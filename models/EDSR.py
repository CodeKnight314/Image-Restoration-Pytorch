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
        return x + self.residual_scaler * self.res_block(x)
    
class EDSR_SingleScale(nn.Module): 
    def __init__(self, input_channels, residual_channels, output_channels, num_resblocks, upscale_factor): 
        super().__init__()

        self.feature_extraction = nn.Conv2d(input_channels, residual_channels, kernel_size=3, stride=1, padding=1)

        layers = []
        for i in range(num_resblocks): 
            layers.append(ResBlock(channels=residual_channels, residual_scaler=0.1))
        self.res_backbone = nn.Sequential(*layers)

        if upscale_factor == 2: 
            self.upsample = nn.Sequential(*[nn.Conv2d(residual_channels, residual_channels * 4, kernel_size=3, stride=1, padding=1),
                                          nn.PixelShuffle(2)])
        elif upscale_factor == 4: 
            self.upsample = nn.Sequential(*[nn.Conv2d(residual_channels, residual_channels * 4, kernel_size=3, stride=1, padding=1),
                                            nn.PixelShuffle(2),
                                            nn.Conv2d(residual_channels, residual_channels * 4, kernel_size=3, stride=1, padding=1), 
                                            nn.PixelShuffle(2)])
            
        self.feature_reconstruction = nn.Conv2d(residual_channels, output_channels,kernel_size=3, stride=1, padding=1)

    def forward(self, x): 
        fe_output = self.feature_extraction(x)
        res_output = self.res_backbone(fe_output)
        up_output = self.upsample(res_output)
        fc_output = self.feature_reconstruction(up_output)

        return x + fc_output
    
def get_EDSR(input_channels = 3, residual_channels = 64, output_channels = 3, num_resblocks = 16, up_scale_factor = 2): 
    return EDSR_SingleScale(input_channels=input_channels, residual_channels=residual_channels, output_channels=output_channels, num_resblocks=num_resblocks, upscale_factor=up_scale_factor)

