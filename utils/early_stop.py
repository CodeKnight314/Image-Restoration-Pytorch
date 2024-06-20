import torch
import torch.nn as nn
import os

class EarlyStopMechanism:
    def __init__(self, metric_threshold: float, mode: str = 'min', grace_threshold: int = 10, save_path: str = 'checkpoints'):
        """
        Initializes the EarlyStopMechanism.

        Args:
            metric_threshold (float): The relative change threshold for considering an improvement.
            mode (str): Mode of comparison, 'min' for minimizing and 'max' for maximizing the metric. Default is 'min'.
            grace_threshold (int): Number of iterations to wait for an improvement before stopping. Default is 10.
            save_path (str): Directory path to save the best model. Default is 'checkpoints'.
        """
        self.metric_threshold = metric_threshold
        self.mode = mode
        self.grace_threshold = grace_threshold
        self.save_path = save_path

        self.best_metric = float("inf") if mode == 'min' else float("-inf")
        self.best_iteration = 0
        self.current_iteration = 0

    def step(self, model: nn.Module, metric: float):
        """
        Updates the mechanism with the latest metric and saves the model if there's an improvement.

        Args:
            model (nn.Module): The model being trained.
            metric (float): The latest metric value to evaluate.
        """
        self.current_iteration += 1

        if self.mode == 'min':
            relative_change = (self.best_metric - metric) / self.best_metric
            improve_condition = relative_change >= self.metric_threshold
        else:
            relative_change = (metric - self.best_metric) / self.best_metric
            improve_condition = relative_change >= self.metric_threshold

        if improve_condition:
            self.best_metric = metric
            self.best_iteration = self.current_iteration
            self.save_model(model)

    def save_model(self, model: nn.Module):
        """
        Saves the model's state dictionary.

        Args:
            model (nn.Module): The model to save.
        """
        os.makedirs(self.save_path, exist_ok=True)
        save_path = os.path.join(self.save_path, f"Epoch_{self.current_iteration}_best_model.pth")
        torch.save(model.state_dict(), save_path)

    def check(self) -> bool:
        """
        Checks if early stopping criteria are met.

        Returns:
            bool: True if early stopping should be applied, False otherwise.
        """
        if self.current_iteration - self.best_iteration >= self.grace_threshold:
            print(f"[INFO] Early Stopping Mechanism engaged. Last improvement was at Epoch {self.best_iteration}.")
            return True
        return False

    def reset(self):
        """
        Resets the early stopping mechanism.
        """
        self.best_metric = float("inf") if self.mode == 'min' else float("-inf")
        self.best_iteration = 0
        self.current_iteration = 0
