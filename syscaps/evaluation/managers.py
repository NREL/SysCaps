import torch
from syscaps.evaluation.metrics import ErrorMetric
from typing import List
from copy import deepcopy



class MetricsManager:
    """A class that keeps track of all metrics (and a scoring rule)for one or more buildings.
    """
    def __init__(self, metrics: List[ErrorMetric] = None):
        """Initializes the MetricsManager.
        
        Args:
            metrics (List[ErrorMetric]): A list of metrics to compute.
        """
        self.metrics = []
        if not metrics is None:
            self.metrics = deepcopy(metrics)
        self.accumulated_unnormalized_loss = 0
        self.total_samples = 0

    def _compute_all(self,
                     y_true: torch.Tensor,
                     y_pred: torch.Tensor,
                     **kwargs) -> None:
        """Computes all metrics and scoring rules for the given batch.
        
        Args:
            y_true (torch.Tensor): A tensor of shape [batch, pred_len, 1] with
                the true values.
            y_pred (torch.Tensor): A tensor of shape [batch, pred_len, 1] with
                the predicted values.
            building_types_mask (torch.Tensor): A tensor of shape [batch] with  
                the building types of each sample.

        """
        if len(self.metrics) > 0:
            for metric in self.metrics:
                metric(y_true, y_pred)
        
    def _compute_scoring_rule(self, 
                            true_continuous,
                            true_categories,
                            y_distribution_params,
                            centroids,
                            building_types_mask) -> None:
        raise NotImplementedError()

    def _update_loss(self, loss, sample_size):
        """Updates the accumulated loss and total samples."""
        self.accumulated_unnormalized_loss += (loss * sample_size)
        self.total_samples += sample_size


    def summary(self, with_loss=False):
        """Return a summary of the metrics for the dataset.
        
        A summary maps keys to objects of type ErrorMetric or ScoringRule.
        """
        summary = {}

        if len(self.metrics) > 0:
            for metric in self.metrics:
                if not metric.UNUSED_FLAG:
                    metric.calculate()
                    summary[metric.name] = metric


        if with_loss and self.total_samples > 0:
            summary['loss'] = self.accumulated_unnormalized_loss / self.total_samples
        return summary

    def reset(self, loss: bool = True) -> None:
        """Reset the metrics."""
        for metric in self.metrics:
            metric.reset()
        if loss:
            self.accumulated_unnormalized_loss = 0
            self.total_samples = 0


    def __call__(self, 
                 y_true: torch.Tensor,
                 y_pred: torch.Tensor,
                 **kwargs):
        """Compute metrics for a batch of predictions.
        
        Args:
            y_true (torch.Tensor): The true (unscaled) load values. (continuous)
                shape is [batch_size, pred_len, 1]
            y_pred (torch.Tensor): The predicted (unscaled) load values. (continuous)
                shape is [batch_size, pred_len, 1]

        Keyword args:
            y_categories (torch.Tensor): The true load values. (quantized)
            y_distribution_params (torch.Tensor): logits, Gaussian params, etc.
            centroids (torch.Tensor): The bin values for the quantized load.
            loss (torch.Tensor): The loss for the batch.
        """
      
        self._compute_all(y_true, y_pred, **kwargs)
        if 'loss' in kwargs:
            batch_size, pred_len, _ = y_true.shape
            self._update_loss(kwargs['loss'], batch_size * pred_len)