import numpy as np
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stops the training if the validation metric doesn't improve after a given patience.
    Tailored for workflows where model checkpointing is handled externally.
    """
    def __init__(self, patience=7, verbose=False, delta=0.0, maximize=True):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Default: 7
            verbose (bool): If True, logs a message for each patience increment.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.0
            maximize (bool): If True, a higher metric is considered better (e.g., C-index, Accuracy).
                             If False, a lower metric is considered better (e.g., negative loss).
                             Default: True
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.maximize = maximize

    def __call__(self, val_metric):
        """
        Args:
            val_metric (float): The current epoch's validation metric.
        """
        # Align logic so that a higher 'score' always means a better model
        score = val_metric if self.maximize else -val_metric

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
