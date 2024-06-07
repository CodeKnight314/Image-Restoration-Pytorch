import torch 
import os 

def count_folders_with_prefix(directory, prefix):
    count = 0
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name.startswith(prefix):
                count += 1
    return count

device = "cuda" if torch.cuda.is_available() else "cpu"

clean_image_dir = ""

degraded_image_dir = ""

image_height = 128 
image_width = 128 

lr = 1e-4 
weight_decay = 1e-3

model_pth = None 

warm_up_phase = 15
epoch = 100

base_path = ""
prefix = "Trial"
output_dir = os.path.join(base_path, f"{prefix}_{count_folders_with_prefix(base_path, prefix)+1}")

log_output_dir = os.path.join(output_dir, "log_outputs")

save_pth = os.path.join(output_dir, "saved_weights")

heatmaps = os.path.join(output_dir, "heatmaps")

def main(): 
    if not os.path.exists(log_output_dir): 
        os.makedirs(log_output_dir)
        print("[INFO] Creating Log output directory.")

    if not os.path.exists(save_pth): 
        os.makedirs(save_pth)
        print("[INFO] Creating save path directory.")
    
    if not os.path.exists(heatmaps):
        os.makedirs(heatmaps)
        print("[INFO] Creating Heat map directory.")

if __name__ == "__main__": 
    main()