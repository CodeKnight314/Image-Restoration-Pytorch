import os
import torch 
import torch.nn as nn 
from tqdm import tqdm 
from utils.log_writer import LOGWRITER 
from utils.visualization import *
from loss import PSNR
import configs

class BaseTrainerIR(): 
    def __init__(self):
        super().__init__()
    
    def train_step(self, model, optimizer, data, criterion):
        """
        Perform a single training step.
        """
        optimizer.zero_grad()
    
        clean_img, degraded_img = data
        
        sr_img = model(degraded_img)
        
        loss = criterion(clean_img, sr_img)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def eval_step(self, model, data, criterion, criterion_psnr): 
        """
        Perform a single evaluation step.
        """
        clean_img, degraded_img = data

        sr_img = model(degraded_img)

        loss = criterion(clean_img, sr_img)

        psnr = criterion_psnr(clean_img, sr_img)

        return loss.item(), psnr.item()
    
    def train_model(self, model, train_dl, valid_dl, optimizer, lr_scheduler, epochs, warmup, log_writer: LOGWRITER): 
        """
        Train the model.
        """
        best_loss = float("inf")

        criterion = nn.MSELoss()
        criterion_psnr = PSNR()

        for epoch in range(1, epochs + 1):
            model.train()
            total_train_loss = 0.0

            for i, data in tqdm(enumerate(train_dl), desc=f"Training Epoch {epoch}/{epochs}"):
                tr_loss = self.train_step(model=model, criterion=criterion, data=data, optimizer=optimizer)
                total_train_loss += tr_loss

            avg_train_loss = total_train_loss / len(train_dl)
            avg_valid_loss, avg_valid_psnr = self.evaluate_model(model, valid_dl, criterion, criterion_psnr)

            if lr_scheduler and epoch > warmup:
                lr_scheduler.step()

            log_writer.write(epoch=epoch, avg_train_loss=avg_train_loss, avg_valid_loss=avg_valid_loss, psnr=avg_valid_psnr)

            if avg_valid_loss < best_loss:
                best_loss = avg_valid_loss
                torch.save(model.state_dict(), os.path.join(configs.save_pth, f'best_model_EPOCH_{epoch}.pth'))

    def evaluate_model(self, model, test_loader, criterion, criterion_psnr): 
        """
        Evaluate the model.
        """
        model.eval()
        
        total_loss = 0.0 
        total_psnr = 0.0
        
        with torch.no_grad(): 
            for i, data in tqdm(enumerate(test_loader), desc="Validating Epoch"):
                loss, psnr = self.eval_step(model, data, criterion, criterion_psnr)
                total_loss += loss
                total_psnr += psnr

        avg_loss = total_loss / len(test_loader)   
        avg_psnr = total_psnr / len(test_loader)

        return avg_loss, avg_psnr
