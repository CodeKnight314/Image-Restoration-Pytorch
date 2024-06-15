import torch 
import torch.nn as nn 
import torch.nn.functional as F

class ResidualDenseBlocks(nn.Module):
    def __init__(self, channels, num_layers): 
        super().__init__() 

        self.num_layers = num_layers
        self.layers = nn.ModuleList() 
        for i in range(num_layers): 
            self.layers.append(nn.Conv2d(channels * (i + 1), channels, kernel_size=3, stride=1, padding=1))
        
        self.reconstruction = nn.Conv2d(channels * (num_layers + 1), channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x 
        concat = x
        for i in range(self.num_layers):
            out = F.relu(self.layers[i](concat))
            concat = torch.cat([concat, out], dim=1)
        output = self.reconstruction(concat)
        output = output + residual
        return output