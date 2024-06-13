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
    def __init__(self, constants = (0.0001, 0.0001)):
        super().__init__() 

        self.C1 = constants[0]
        self.C2 = constants[1]

    def channel_calculation(self, x, y):
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        
        variance_x = torch.var(x, unbiased=False)
        variance_y = torch.var(y, unbiased=False)
        
        covariance_xy = torch.mean((x - mean_x) * (y - mean_y))
        
        SSIM = ((2 * mean_x * mean_y + self.C1) * (2 * covariance_xy + self.C2)) / ((mean_x ** 2 + mean_y ** 2 + self.C1) * (variance_x + variance_y + self.C2))
        
        return SSIM
    
    def channel_splits(self, x, y):
        assert x.shape[1] == y.shape[1], "[ERROR] X and Y have different number of channels."

        x_tensors = []
        y_tensors = []

        for i in range(x.shape[1]+1):
            x_tensors.append(x[:, i, :, :])
            y_tensors.append(y[:, i, :, :])
        
        return x_tensors, y_tensors
    
    def forward(self, clean_img, sr_img):
        x_tensors, y_tensors = self.channel_splits(clean_img, sr_img)

        SSIM_values = 0.0
        for x, y in zip(x_tensors, y_tensors):
            SSIM_values += self.channel_calculation(x, y)
        
        return 1 - SSIM_values / len(x_tensors)
        
    
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