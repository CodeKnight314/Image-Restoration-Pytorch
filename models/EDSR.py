import torch 
import torch.nn as nn 

class ResBlock(nn.Module): 
    def __init__(self, channels, residual_scaler): 
        super().__init__()

        self.res_block = nn.Sequential(nn.Conv2d(channels, channels), 
                                        nn.ReLU(),
                                        nn.Conv2d(channels, channels))
        
        self.residual_scaler = residual_scaler

    def forward(self, x): 
        return x + self.residual_scaler * self.res_block(x)
    
