import torch 
import configs
import argparse 
import json
from dataset import load_dataset
from loss import MSE_Loss
from models import Restormer, DnCNN
import torch.optim as opt
import torch.multiprocessing as mp
from utils.log_writer import LOGWRITER

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train a model on CIFAR-10')
    parser.add_argument('--model', type=str, required=True, choices=['Restormer', 'DnCNN'], help='Model name')
    parser.add_argument('--model_save_path', type=str, help='Path to save or load model weights')
    parser.add_argument('--root_dir', type=str, required=True, help="Root directory to Dataset. Must contain a train and test folder in root directory.")
    parser.add_argument('--config_file', type=str, required=True, default='config.json', help='Path to configuration file')

    args = parser.parse_args()
    
    model_config = load_config(args.config_file)

    # Declaring DataLoaders
    train_dl = load_dataset(root_dir=args.root_dir, 
                            patch_size=model_config.get('patch_size'),
                            batch_size=model_config.get('batch_size'),
                            mode="train")
    
    valid_dl = load_dataset(root_dir=args.root_dir, 
                            patch_size=model_config.get('patch_size'), 
                            batch_size=model_config.get('batch_size'),
                            mode="val")
    
    print(f"[INFO] Training Dataloader loaded with {len(train_dl)} batches.")
    print(f"[INFO] Validation Dataloader loaded with {len(valid_dl)} batches.")

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
        trainer = DnCNN.DnCNNTrainer()
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
            
    if model_config.get("optimizer") == 'AdamW':
        optimizer = opt.AdamW(model.parameters(), lr=model_config.get("lr"), weight_decay=model_config.get("weight decay"))
    elif model_config.get("optimizer") == 'SGD':
        optimizer = opt.SGD(model.parameters(), lr=model_config.get("lr"), weight_decay=model_config.get("weight decay"), momentum=0.9)
    print(f"[INFO] Optimizer loaded with learning rate: {model_config.get('lr')}.")

    if model_config.get("scheduler") == 'CosineAnnealingLR':
        scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_config.get("t_max"), eta_min=model_config.get("eta_min"))
    elif model_config.get("scheduler") == 'StepLR':
        scheduler = opt.lr_scheduler.StepLR(optimizer, step_size=model_config.get("step_size"), gamma=model_config.get("gamma"))
    print(f"[INFO] {model_config.get('scheduler')} Scheduler loaded.")

    logger = LOGWRITER(output_directory=configs.log_output_dir, total_epochs=model_config.get('epochs'))
    print(f"[INFO] Log writer loaded and binded to {configs.log_output_dir}")
    print(f"[INFO] Total epochs: {model_config.get('epochs')}")
    print(f"[INFO] Warm Up Phase: {model_config.get('warmup')} epochs")

    configs.main()

    model.to(configs.device)

    trainer.train_model(model=model,
                        train_dl=train_dl, 
                        valid_dl=valid_dl, 
                        optimizer=optimizer, 
                        lr_scheduler=scheduler, 
                        epochs=model_config.get('epochs'), 
                        warmup=model_config.get('warmup'),
                        log_writer=logger)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()