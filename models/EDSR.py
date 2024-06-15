import torch 
import torch.nn as nn 

class ResBlock(nn.Module): 
    def __init__(self, channels, residual_scaler): 
        super().__init__()

        self.res_block = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3), 
                                        nn.ReLU(),
                                        nn.Conv2d(channels, channels))
        
        self.residual_scaler = residual_scaler

    def forward(self, x): 
        out = self.res_block(x)
        out = self.residual_scaler * out 
        out = x + out
        return out
    
class UpsampleBlock(nn.Module):
    def __init__(self, channels, scale_factor): 
        super().__init__() 

        self.conv = nn.Conv2d(channels, channels * (scale_factor) ** 2, kernel_size=3, stride=1, padding=1)
        self.up = nn.PixelShuffle(scale_factor)
    
    def forward(self, x): 
        out = self.conv(x)
        out = self.up(x)
        return out
    
class EDSR_SingleScale(nn.Module): 
    def __init__(self, input_channels, residual_channels, output_channels, num_resblocks, upscale_factor): 
        super().__init__()

        self.feature_extraction = nn.Conv2d(input_channels, residual_channels, kernel_size=3, stride=1, padding=1)

        layers = []
        for i in range(num_resblocks): 
            layers.append(ResBlock(channels=residual_channels, residual_scaler=0.1))
        self.res_backbone = nn.Sequential(*layers)

        self.upsample = UpsampleBlock(residual_channels, upscale_factor)
            
        self.feature_reconstruction = nn.Conv2d(residual_channels, output_channels,kernel_size=3, stride=1, padding=1)

    def forward(self, x): 
        fe_output = self.feature_extraction(x)
        res_output = self.res_backbone(fe_output)
        up_output = self.upsample(res_output)
        fc_output = self.feature_reconstruction(up_output)

        return x + fc_output

class EDSR_MultiScale(nn.Module): 
    def __init__(self, input_channels, residual_channels, output_channels, num_resblocks): 
        super().__init__() 

        self.feature_extraction = nn.Conv2d(input_channels, residual_channels, kernel_size=3, stride=1, padding=1)

        layers = [] 
        for i in range(num_resblocks): 
            layers.append(ResBlock(residual_channels=residual_channels, residual_scaler=0.1))
        self.res_backbone = nn.Sequential(*layers)

        self.upsample_2x = UpsampleBlock(residual_channels, 2)
        self.upsample_3x = UpsampleBlock(residual_channels, 3)
        self.upsample_4x = UpsampleBlock(residual_channels, 4)

        self.feature_reconstruction = nn.Conv2d(residual_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, scale_factor): 
        fe_output = self.feature_extraction(x)
        res_output = self.res_backbone(fe_output)

        if scale_factor == 2: 
            up_output = self.upsample_2x(res_output)
        elif scale_factor == 3: 
            up_output = self.upsample_3x(res_output)
        else: 
            up_output = self.upsample_4x(res_output)

        out = self.feature_reconstruction(up_output)
        return x + out
    
def get_EDSR(input_channels = 3, residual_channels = 64, output_channels = 3, num_resblocks = 16, up_scale_factor = 2): 
    """
    """
    return EDSR_SingleScale(input_channels=input_channels, residual_channels=residual_channels, output_channels=output_channels, num_resblocks=num_resblocks, upscale_factor=up_scale_factor)

