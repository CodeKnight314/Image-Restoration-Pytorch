import torch 
import torch.nn as nn 
from utils.log_writer import LOGWRITER
from tqdm import tqdm

def evaluate(model, test_loader, criterion, crtierion_psnr, criterion_ssim): 
    model.eval() 
    
    total_loss = 0.0 
    total_psnr_loss = 0.0 
    total_ssim_loss = 0.0
    
    with torch.no_grad(): 

        for i, data in tqdm(enumerate(test_loader)):
            clean_img, degraded_img = data 

            sr_img = model(degraded_img)

            loss = criterion(clean_img, sr_img)
            psnr = crtierion_psnr(clean_img, sr_img)
            ssim = criterion_ssim(clean_img, sr_img)

    avg_loss = total_loss / len(test_loader)
    avg_psnr_loss = total_psnr_loss / len(test_loader)
    avg_ssim_loss = total_ssim_loss / len(test_loader)    

        