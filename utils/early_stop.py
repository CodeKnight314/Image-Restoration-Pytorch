import torch
import torch.nn as nn 
import configs 
import os

class EarlyStopMechanism():
    def __init__(self, metric_threshold, mode, grace_threshold):
        self.metric_threshold = metric_threshold 
        self.mode = mode
        self.grace_threhold = grace_threshold

        self.best_loss = float("inf")
        self.best_iteration = 0

        self.current_loss = 0.0
        self.current_iteration = 0

    def step(self, model, loss):
        self.current_loss = loss 
        if self.current_loss < self.best_loss - self.metric_threshold:
            self.best_loss = self.current_loss
            self.best_iteration = self.current_iteration 
            self.save_model(model)
        self.current_iteration+=1

    def save_model(self, model : nn.Module):
        torch.save(model.state_dict(), 
                   os.path.join(configs.save_pth, f"Epoch_{self.current_iteration}_best_model.pth"))
    
    def check(self, model):
        if self.current_iteration - self.best_iteration >= self.grace_threhold:
            print(f"[INFO] Early Stopping Mechanism engaged. Last loss update was Epoch {self.best_iteration+1}")
            return True
        else: 
            return False
