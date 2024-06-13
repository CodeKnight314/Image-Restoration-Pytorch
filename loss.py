import torch
import torch.nn as nn
import torch.nn.functional as F

class PSNR(nn.Module):
    def __init__(self, maximum_pixel_value): 
        super().__init__() 
        self.me_loss = MSE_Loss()
        self.max_value = maximum_pixel_value

    def forward(self, clean_img, sr_img): 
        mse_loss = self.me_loss(clean_img, sr_img) 
        return 10 * torch.log10(self.max_value ** 2 / mse_loss)

class SSIM(nn.Module):
    def __init__(self):
        super().__init__() 
    
    def forward(self, x): 
        return x
    
class GradientPriorLoss(nn.Module):
    def __init__(self):
        super(GradientPriorLoss, self).__init__()
        self.func = nn.L1Loss()

    def forward(self, clean_img, sr_img):
        map_clean= self.gradient_map(clean_img)
        map_sr = self.gradient_map(sr_img)
        return self.func(map_clean, map_sr)

    def gradient_map(self, x):
        _, _, h_x, w_x = x.size()

        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:] 
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]

        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2)+1e-6, 0.5)
        return xgrad
    
class MSE_Loss(nn.Module):
    def __init__(self): 
        super.__init__()

    def forward(self, x, y): 
        difference = x - y 
        return torch.mean(difference ** 2)