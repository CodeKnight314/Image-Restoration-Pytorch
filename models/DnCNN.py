import torch 
import torch.nn as nn 
from base_model import BaseModelIR

class DnCNN(BaseModelIR): 
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers):
        super().__init__() 
        
        self.feature_extraction = nn.Sequential(nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1), 
                                                nn.ReLU())
        
        layers = [] 
        for _ in range(num_layers):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU())

        self.conv_backbone = nn.Sequential(*layers)

        self.feature_reconstruction = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x): 
        out = self.feature_extraction(x)
        out = self.conv_backbone(out)
        out = self.feature_reconstruction(out)
        return out 
    
def get_DnCNN(input_channels = 3, hidden_channels = 64, output_channels = 3, num_layers = 20): 
    return DnCNN(input_channels=input_channels, hidden_channels=hidden_channels, output_channels=output_channels, num_layers=num_layers)