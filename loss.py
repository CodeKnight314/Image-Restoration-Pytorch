import torch
import torch.nn as nn

class PSNR():
    def __init__(self, maximum_pixel_value): 
        super().__init__() 
        self.max_value = maximum_pixel_value

    def forward(self, clean_img, degraded_img): 
        mse_loss = nn.MSELoss(clean_img, degraded_img) #Maybe implement this by hand for fun?
        return 10 * torch.log10(self.max_value ** 2 / mse_loss)
