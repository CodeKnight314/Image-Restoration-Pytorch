import torch
import torch.nn as nn 
import configs 
import os

class EarlyStopMechanism:
    def __init__(self, metric_threshold, mode='min', grace_threshold=10, save_path='checkpoints'):
        self.metric_threshold = metric_threshold
        self.mode = mode
        self.grace_threshold = grace_threshold
        self.save_path = save_path

        self.best_metric = float("inf") if mode == 'min' else float("-inf")
        self.best_iteration = 0
        self.current_iteration = 0

    def step(self, model, metric):
        self.current_iteration += 1
        improve_condition = (metric < self.best_metric - self.metric_threshold) if self.mode == 'min' else (metric > self.best_metric + self.metric_threshold)
        
        if improve_condition:
            self.best_metric = metric
            self.best_iteration = self.current_iteration
            self.save_model(model)

    def save_model(self, model: nn.Module):
        torch.save(model.state_dict(), os.path.join(self.save_path, f"Epoch_{self.current_iteration}_best_model.pth"))

    def check(self):
        if self.current_iteration - self.best_iteration >= self.grace_threshold:
            print(f"[INFO] Early Stopping Mechanism engaged. Last improvement was at Epoch {self.best_iteration}.")
            return True
        return False

    def reset(self):
        self.best_metric = float("inf") if self.mode == 'min' else float("-inf")
        self.best_iteration = 0
        self.current_iteration = 0