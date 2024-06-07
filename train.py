import torch 
import torch.nn as nn 
import configs
from tqdm import tqdm
import os

def train_step(model, criterion, data, optimizer): 
    """
    """
    optimizer.zero_grad()
    clean_img, degraded_img = data 
    sr_img = model(degraded_img)
    loss = criterion(clean_img, sr_img)

    optimizer.step() 

    return loss.item()

def valid_step(model, criterion, data, criterion_psnr):
    """
    """
    clean_img, degraded_img = data 
    sr_img = model(degraded_img)
    loss = criterion(clean_img, sr_img)
    psnr = criterion_psnr(clean_img, sr_img)

    return loss.item(), psnr.item()

def train(model, criterion, criterion_psnr, train_dl, valid_dl, optimizer, scheduler, epochs): 
    """
    """
    best_loss = float("inf")
    model.to(configs.device)

    for epoch in range(1, epochs+1):

        model.train() 
        total_train_loss = 0.0
        for i, data in tqdm(enumerate(train_dl)):
            tr_loss = train_step(model, criterion=criterion, data=data, optimizer=optimizer)
            total_train_loss+=tr_loss

        model.eval()   
        total_valid_loss = 0.0 
        total_valid_psnr = 0.0     
        with torch.no_grad(): 
            for i, data in tqdm(enumerate(valid_dl)):
                valid_loss, valid_psnr = valid_step(model=model, criterion=criterion, data=data, criterion_psnr=criterion_psnr)
                total_valid_loss+=valid_loss 
                total_valid_psnr+=valid_psnr
            
        avg_train_loss = total_train_loss / len(train_dl)
        avg_valid_loss = total_valid_loss / len(valid_dl)
        avg_valid_psnr = total_valid_psnr / len(valid_dl)

        if avg_valid_loss < best_loss: 
            best_loss = avg_valid_loss
            save_path = os.path.join(configs.save_path, f"Best_model_{epoch}.pth")
            torch.save(model.state_dict(), save_path)

        if epoch > configs.warm_up_phase: 
            scheduler.step()

def main(): 
    """
    """

if __name__ == "__main__": 
    main()
