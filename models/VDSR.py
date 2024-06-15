import torch 
import math
import torch.nn as nn 
from utils.complexity_measure import *

class VDSR(nn.Module): 
    def __init__(self, input_channels, output_channels, num_layers):
        super().__init__() 

        self.feature_extraction = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1), nn.ReLU())

        layers = [] 
        for i in range(num_layers): 
            layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
        
        self.cnn_backbone = nn.Sequential(*layers)

        self.feature_recontruction = nn.Sequential(nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1), nn.ReLU())

    def forward(self, x): 
        return x + self.feature_recontruction(self.cnn_backbone(self.feature_extraction(x)))
    
    def weight_initialization(self): 
        for module in self.modules(): 
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)
    
def get_VDSR(input_channels=3, output_channels=3, num_layers=6): 
    """
    """
    return VDSR(input_channels=input_channels, output_channels=output_channels, num_layers=num_layers)

if __name__ == "__main__": 
    vdsr = get_VDSR()

    input_size = (1, 3, 256, 256)

    vdsr_gflops = count_model_flops(vdsr, input_size=input_size)

    print(f"[INFO] MDSR GFLOPS: {vdsr_gflops}")