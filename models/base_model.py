import torch 
import torch.nn as nn 
from tqdm import tqdm 
from utils.log_writer import * 
from utils.visualization import *

class BaseModelIR(nn.Module): 
    def __init__(self, model): 
        super().__init__()
    
    def train_step(self, optimizer, data, criterion):
        """
        """
        optimizer.zero_grad()
    
        clean_img, degraded_img = data
        
        sr_img = self(degraded_img)
        
        loss = criterion(clean_img, sr_img)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def eval_step(self, data, criterion): 
        """
        """
        clean_img, degraded_img = data

        sr_img = self(degraded_img)

        loss = criterion(clean_img, sr_img)

        return loss.item()
    
    def train(self, train_dl, valid_dl, optimizer, criterion, lr_scheduler, epochs, log_writer : LOGWRITER): 
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            self.train()
            total_train_loss = 0.0

            for i, data in tqdm(enumerate(train_dl), total=len(train_dl)):
                tr_loss = self.train_step(criterion=criterion, data=data, optimizer=optimizer)
                total_train_loss += tr_loss

            avg_train_loss = total_train_loss / len(train_dl)
            avg_valid_loss = self.evaluate(valid_dl, criterion, criterion, criterion)

            log_writer.write(epoch=epoch, avg_train_loss=avg_train_loss, avg_valid_loss=avg_valid_loss)

    def evaluate(self, test_loader, criterion): 
        self.eval() 
        
        total_loss = 0.0 
        
        with torch.no_grad(): 

            for i, data in tqdm(enumerate(test_loader)):
                loss = self.eval_step(data, criterion)
                total_loss+=loss

        avg_loss = total_loss / len(test_loader)   

        return avg_loss