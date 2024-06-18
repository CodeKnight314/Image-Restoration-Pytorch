import torch 
import torch.nn as nn 
from utils.log_writer import LOGWRITER
from tqdm import tqdm
import configs
from utils.visualization import *

def evaluate(model, test_loader, criterion, criterion_psnr, criterion_ssim, log_writer : LOGWRITER):
    """
    """ 
    model.eval() 
    
    total_loss = 0.0 
    total_psnr_loss = 0.0 
    total_ssim_loss = 0.0
    
    with torch.no_grad(): 

        for i, data in tqdm(enumerate(test_loader)):
            clean_img, degraded_img = data 

            sr_img = model(degraded_img)

            loss = criterion(clean_img, sr_img)
            
            clean_img_YCbCr = rgb_to_ycbcr(clean_img)
            clean_img_Y = clean_img_YCbCr[:, 0, :, :]
            
            sr_img_YCbCr = rgb_to_ycbcr(sr_img)
            sr_img_Y = sr_img_YCbCr[:, 0, :, :]
            
            psnr = criterion_psnr(clean_img_Y, sr_img_Y)
            
            ssim = criterion_ssim(clean_img, sr_img)

            total_loss+=loss
            total_psnr_loss+=psnr
            total_ssim_loss+=ssim

    avg_loss = total_loss / len(test_loader)
    avg_psnr_loss = total_psnr_loss / len(test_loader)
    avg_ssim_loss = total_ssim_loss / len(test_loader)    

    log_writer.write(epoch=0, avg_loss = avg_loss, avg_psnr_loss=avg_psnr_loss, avg_ssim_loss=avg_ssim_loss)
    
def main(): 
    """
    """
    configs.main() 

if __name__ == "__main__":
    main()


        