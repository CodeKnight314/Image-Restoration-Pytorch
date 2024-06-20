import torch 
import torch.nn as nn 
from utils.log_writer import LOGWRITER
from tqdm import tqdm
import configs
from utils.visualization import *

def evaluate(model: nn.Module, test_loader: torch.utils.data.DataLoader, criterion: nn.Module, criterion_psnr: nn.Module, criterion_ssim: nn.Module, log_writer: LOGWRITER):
    """
    Evaluates the performance of the model on the test dataset.

    Args:
        model (nn.Module): The neural network model being evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): The loss function used for evaluation.
        criterion_psnr (nn.Module): The PSNR criterion for evaluating image quality.
        criterion_ssim (nn.Module): The SSIM criterion for evaluating image quality.
        log_writer (LOGWRITER): An object for logging evaluation metrics.
    """
    model.eval() 
    
    total_loss = 0.0 
    total_psnr_loss = 0.0 
    total_ssim_loss = 0.0
    
    with torch.no_grad(): 
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            clean_img, degraded_img = data 

            sr_img = model(degraded_img)

            loss = criterion(clean_img, sr_img)
            psnr = criterion_psnr(clean_img, sr_img)
            ssim = criterion_ssim(clean_img, sr_img)

            total_loss += loss
            total_psnr_loss += psnr
            total_ssim_loss += ssim

    avg_loss = total_loss / len(test_loader)
    avg_psnr_loss = total_psnr_loss / len(test_loader)
    avg_ssim_loss = total_ssim_loss / len(test_loader)    

    log_writer.write(epoch=0, avg_loss=avg_loss.item(), avg_psnr_loss=avg_psnr_loss.item(), avg_ssim_loss=avg_ssim_loss.item())

def main(): 
    """
    The main function to run the evaluation process.
    """
    configs.main() 

if __name__ == "__main__":
    main()