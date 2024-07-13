import torch 
import torch.nn as nn 
from tqdm import tqdm
import argparse
import json
from dataset import load_dataset
from models import Restormer, DnCNN
from loss import MSE_Loss, PSNR, SSIM

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def evaluate(model: nn.Module, test_dl: torch.utils.data.DataLoader):
    """
    Evaluates the performance of the model on the test dataset.

    Args:
        model (nn.Module): The neural network model being evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): The loss function used for evaluation.
        criterion_psnr (nn.Module): The PSNR criterion for evaluating image quality.
        criterion_ssim (nn.Module): The SSIM criterion for evaluating image quality.
    """
    model.eval() 

    criterion = MSE_Loss()
    criterion_psnr = PSNR()
    criterion_ssim = SSIM()
    
    total_loss = 0.0 
    total_psnr_loss = 0.0 
    total_ssim_loss = 0.0
    
    with torch.no_grad(): 
        for i, data in tqdm(enumerate(test_dl), total=len(test_dl)):
            clean_img, degraded_img = data 

            sr_img = model(degraded_img)

            loss = criterion(clean_img, sr_img)
            psnr = criterion_psnr(clean_img, sr_img)
            ssim = criterion_ssim(clean_img, sr_img)

            total_loss += loss
            total_psnr_loss += psnr
            total_ssim_loss += ssim

    avg_loss = total_loss / len(test_dl)
    avg_psnr_loss = total_psnr_loss / len(test_dl)
    avg_ssim_loss = total_ssim_loss / len(test_dl)
    
    print(f"Image Restoration Performance:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average PSNR: {avg_psnr_loss:.4f}")
    print(f"Average SSIM: {avg_ssim_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train a model on CIFAR-10')
    parser.add_argument('--model', type=str, required=True, choices=['ViT', 'ResNet18', 'ResNet34','HCVIT', 'MobileNet'], help='Model name')
    parser.add_argument('--model_save_path', type=str, help='Path to save or load model weights')
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory to Dataset. Must contain a train and test folder in root directory.")
    parser.add_argument('--config_file', type=str, required=True, default='config.json', help='Path to configuration file')

    args = parser.parse_args()
    
    model_config = load_config(args.config_file)

    test_dl = load_dataset(root_dir=args.root_dir, 
                            patch_size=model_config.get('patch_size'), 
                            batch_size=model_config.get('batch_size'),
                            mode="test")

    print(f"[INFO] Test Dataloader loaded with {len(test_dl)} batches.")

    if args.model == "Restormer": 
        model = Restormer.Restormer(input_channels=model_config.get("input_channels"), 
                                    output_channels=model_config.get("output_channels"), 
                                    channels=model_config.get("channels"),
                                    num_levels=model_config.get("num_levels"), 
                                    num_transformers=model_config.get("num_transformers"),
                                    num_heads=model_config.get("num_heads"),
                                    expansion_factor=model_config.get("expansion_factor"))
        print("[INFO] Restormer model loaded")
    elif args.model == "DnCNN": 
        model = DnCNN.DnCNN(input_channels=model_config.get("input_channels"),
                            hidden_channels=model_config.get("hidden_channels"),
                            output_channels=model_config.get("output_channels"),
                            num_layers=model_config.get("num_layers"))
        print("[INFO] DnCNN model loaded")
    
    if args.model_save_path:
        print("[INFO] Model weights provided. Attempting to load model weights.")
        try:
            model.load_state_dict(torch.load(args.model_save_path), strict=False)
            print("[INFO] Model weights loaded successfully with strict=False.")
        except RuntimeError as e:
            print(f"[WARNING] Runtime error occurred while loading some model weights: {e}")
        except FileNotFoundError as e:
            print(f"[ERROR] File not found error occurred: {e}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while loading model weights: {e}")
    else:
        print("[INFO] No model weights path provided. Training from scratch.")

    evaluate(model=model, test_dl=test_dl)
