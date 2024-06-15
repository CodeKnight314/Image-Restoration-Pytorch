import torch 
import torch.nn as nn 
import math 
import configs
from utils.complexity_measure import * 

class SRCNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__() 

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels[0], kernel_size=9, stride=1, padding=4),
            nn.ReLU()
        )

        self.non_linear_mapping = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        self.feature_reconstruction = nn.Sequential(
            nn.Conv2d(hidden_channels[0], output_channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.feature_reconstruction(self.non_linear_mapping(self.feature_extraction(x)))
    
    def weight_initialization(self):
        for module in self.modules(): 
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

def get_SRCNN(input_channels = 3, hidden_channels = [64, 32], output_channels = 3):
    """
    """
    return SRCNN(input_channels=input_channels, hidden_channels=hidden_channels, output_channels=output_channels).to(configs.device)

if __name__ == "__main__":  
    srcnn = get_SRCNN()

    input_size = (1, 3, 256, 256)
    
    srcnn_gflops = count_model_flops(srcnn, input_size=input_size)

    print(f"[INFO] SRCNN GFLOPS: {srcnn_gflops}")