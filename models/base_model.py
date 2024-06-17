import torch 
import torch.nn as nn 

class BaseModelIR(nn.Module): 
    def __init__(self, model): 
        super().__init__()
    
    def train_step(self, optimizer, data, criterion):
        """
        """
        optimizer.zero_grad()
    
        clean_img, degraded_img = data
        
        sr_img = self(degraded_img)
        
        loss = criterion(clean_img, sr_img)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def eval_step(self, data, criterion): 
        """
        """
        clean_img, degraded_img = data

        sr_img = self(degraded_img)

        loss = criterion(clean_img, sr_img)

        return loss.item()