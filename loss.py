import torch
import torch.nn as nn

class PSNR(nn.Module):
    def __init__(self, maximum_pixel_value): 
        super().__init__() 
        self.me_loss = MSE_Loss()
        self.max_value = maximum_pixel_value

    def forward(self, clean_img, degraded_img): 
        mse_loss = self.me_loss(clean_img, degraded_img) #Maybe implement this by hand for fun?
        return 10 * torch.log10(self.max_value ** 2 / mse_loss)
    
class MSE_Loss(nn.Module):
    def __init__(self): 
        super.__init__()

    def forward(self, x, y): 
        difference = x - y 
        return torch.mean(difference ** 2)
