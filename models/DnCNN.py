import torch.nn as nn 
from .base_model import BaseTrainerIR
from utils.log_writer import LOGWRITER

class DnCNN(nn.Module): 
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers):
        super().__init__() 
        
        self.feature_extraction = nn.Sequential(nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1), 
                                                nn.ReLU())
        
        layers = [] 
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU())

        self.conv_backbone = nn.Sequential(*layers)

        self.feature_reconstruction = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x): 
        residual = x
        out = self.feature_extraction(x)
        out = self.conv_backbone(out)
        out = self.feature_reconstruction(out)
        return residual + out 
    
class DnCNNTrainer(BaseTrainerIR):
    def train_model(self, model, train_dl, valid_dl, optimizer, lr_scheduler, epochs, warmup, log_writer: LOGWRITER):
        return super().train_model(model, train_dl, valid_dl, optimizer, lr_scheduler, epochs, warmup, log_writer)
    
    def evaluate_model(self, model, test_loader, criterion, criterion_psnr):
        return super().evaluate_model(model, test_loader, criterion, criterion_psnr)