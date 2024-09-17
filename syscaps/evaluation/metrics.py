from typing import Callable
import torch 


class ErrorMetric:
    """A class that represents an error metric.  
    """
    def __init__(self, name: str,
                       simrun_or_stock: str,
                       hourly_or_annual: str,
                       error_function: Callable,
                       normalize: bool,
                       sqrt: bool):
        """
        Args:
            name (str): The name of the metric.
            simrun_or_stock (str): 'simrun', 'stock'; average per run of simulation or summed over sim runs
            hourly_or_annual (str): 'hourly' , 'annual' ; time dimension aggregation (average/sum)
            error_function (Callable): absolute_error, squared_error, bias_error
            normalize (bool): Whether to normalize the error.
            sqrt (bool): Whether to take the square root of the error (for root mean errors)
        """
        self.name = name
        self.simrun_or_stock = simrun_or_stock
        self.hourly_or_annual = hourly_or_annual
        self.error_function = error_function
        self.normalize = normalize
        self.sqrt = sqrt

        self.preds = []
        self.targets = []

        self.UNUSED_FLAG = True

    def __call__(self, y_true, y_pred) -> None:
        """
        Args:
            y_true (torch.Tensor): shape [N, *]
            y_pred (torch.Tensor): shape [N, *]
        """
        self.UNUSED_FLAG = False
        self.preds += [y_pred]
        self.targets += [y_true]

    def reset(self) -> None:
        """Reset the metric."""
        self.targets = []
        self.preds = []
        self.value = None
        self.UNUSED_FLAG = True

    def calculate(self) -> None:
        """Calculate the the error metric, populating the value attribute."""
        if self.UNUSED_FLAG:
            # Returning a number >= 0 is undefined,
            # because this metric is unused. -1
            # is a flag to indicate this.
            return
        
        # When we concatenate errors and global values
        # we want shape errors to be shape]
        # and global values to be 1D
        # if self.errors[0].dim() == 1:
        #     self.errors = [e.unsqueeze(0) for e in self.errors]
        # if self.global_values[0].dim() == 0:
        #     self.global_values = [g.unsqueeze(0) for g in self.global_values]

        # concatenate along the first dim
        all_preds = torch.concatenate(self.preds,0)
        all_targets = torch.concatenate(self.targets,0)

        # make sure tensors are [# simruns, sequence length]
        all_preds = all_preds.reshape(all_preds.shape[0], all_preds.shape[1])
        all_targets = all_targets.reshape(all_targets.shape[0], all_targets.shape[1])
            
        if self.hourly_or_annual == 'annual': # sum over hours
            all_preds = torch.sum(all_preds, dim=1)
            all_targets = torch.sum(all_targets, dim=1)
        
        if self.simrun_or_stock == 'stock': # sum over simruns
            all_preds = torch.sum(all_preds, dim=0)
            all_targets = torch.sum(all_targets, dim=0)

        # apply function 
        all_errors = self.error_function(all_preds - all_targets)
        
        # take mean of remaining dimensions
        mean = torch.mean(all_errors)

        # for root mean error
        if self.sqrt:
            mean = torch.sqrt(mean)

        if self.normalize:
            mean = mean / torch.mean(all_targets)
        self.value = mean
    
    
################## METRICS ##################

def absolute_error(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x)

def squared_error(x: torch.Tensor) -> torch.Tensor:
    return torch.square(x)
 
def bias_error(x: torch.Tensor) -> torch.Tensor:
    return x