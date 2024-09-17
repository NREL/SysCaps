import abc
from typing import Dict, Union
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict
from datetime import datetime, timedelta
from syscaps.evaluation import metrics
from syscaps.evaluation.managers import MetricsManager


TIMESTAMPS = [
        datetime(2018, 1, 1, 0, 0) + k * timedelta(minutes=60) for k in range(8760)
    ]
TIMESTAMPS = [t for t in TIMESTAMPS if t != datetime(2018, 3, 11, 2, 0)]


class Tracker(metaclass=abc.ABCMeta):
    '''Base Tracker class'''
    

    def __init__(self, groups: Dict) -> None:
        '''Init method for base class
        
        Args:
            groups (Dict): dictionary of group name and building ids pairs
        '''
        super().__init__()
        self.groups = groups

    def update(self, values: Union[torch.Tensor, tuple], building_ids: np.array) -> None:
        for group in self.groups:
            mask = np.isin(building_ids, self.groups[group], assume_unique=True)
            if mask.any():
                self.update_(group, mask, values, building_ids)
    
    def update_(self, group: str, mask: np.array, values: Union[torch.Tensor, tuple], building_ids: np.array) -> None:
        '''Update tracked values based on building ids'''
        raise NotImplementedError
    
    def get(self, group: str) -> Union[np.array, float, int]:
        '''Get tracked values based on group name'''
        raise NotImplementedError

'''
Tracks buildings seen in the given dataset
'''
class BuildingsTracker(Tracker):
    def __init__(self, groups: Dict) -> None:
        '''Init method for base class
        
        Args:
            groups (Dict): dictionary of group name and building ids pairs
        '''
        super(BuildingsTracker, self).__init__(groups)
        self.buildings = defaultdict(set)

    def update_(self, group: str, mask: np.array, values: Union[torch.Tensor, tuple], building_ids: np.array) -> None:
        self.buildings[group].update(building_ids[mask])

    def get(self, group: str) -> int:
        return len(self.buildings[group])
    
class YearlyTracker(Tracker):
    '''Tracker for average yearly load history

        Args:
            groups (Dict): dictionary of group name and building ids pairs
    '''
    def __init__(self, groups: Dict) -> None:
        super(YearlyTracker, self).__init__(groups)
        self.mean = {} # current mean values
        self.nums = {} # total number samples

    def update_(self, group: str, mask: np.array, values: Union[torch.Tensor, tuple], building_ids: np.array) -> None:
        if group not in self.mean:
            self.mean[group] = values[mask, :].mean(dim=0)
            self.nums[group] = mask.astype(int).sum()
        else:
            val = self.mean[group] * self.nums[group] + values[mask, :].sum(dim=0)
            self.nums[group] += mask.astype(int).sum()
            self.mean[group] = val / self.nums[group]

    def get(self, group: str) -> np.array:
        return self.mean[group].cpu().numpy()
    
class DailyTracker(Tracker):
    '''Tracker for average daily load history

        Args:
            groups (Dict): dictionary of group name and building ids pairs
    '''
    
    def __init__(self, groups: Dict) -> None:
        super(DailyTracker, self).__init__(groups)
        self.sums = {} # current mean values
        self.nums = {} # total number samples
        hours = [[] for _ in range(24)]
        for i, t in enumerate(TIMESTAMPS):
            hours[t.hour].append(i)
        self.hours = [np.array(h) for h in hours]

    def update_(self, group: str, mask: np.array, values: Union[torch.Tensor, tuple], building_ids: np.array) -> None:
        if group not in self.sums:
            self.sums[group] = np.zeros(24)
            self.nums[group] = np.zeros(24)

        for h in range(24):
            self.sums[group][h] += values[mask, :][:, self.hours[h]].sum()
            self.nums[group][h] += len(self.hours[h]) * mask.astype(int).sum()

    def get(self, group: str) -> np.array:
        return self.sums[group] / self.nums[group]
    
class MetricsTracker(Tracker):
    '''Tracker for nrmse and nmbe'''
    def __init__(self, groups: Dict) -> None:
        super(MetricsTracker, self).__init__(groups)
        self.managers = {}
        for group in groups:
            self.managers[group] = MetricsManager(
                    metrics=[metrics.ErrorMetric('nrmse-stock-annual',  'stock', 'annual', metrics.squared_error, normalize=True, sqrt=True),
                             metrics.ErrorMetric('nmbe-stock-annual', 'stock', 'annual', metrics.bias_error, normalize=True, sqrt=False)]) 
    
    def update_(self, group: str, mask: np.array, values: Union[torch.Tensor, tuple], building_ids: np.array) -> None:
        targets, preds = values
        self.managers[group](
            y_true=targets[mask, :],
            y_pred=preds[mask, :]
        ) 

    def get(self, group: str) -> tuple:
        summary = self.managers[group].summary()
        return summary["nrmse-stock-annual"].value, summary["nmbe-stock-annual"].value
    
