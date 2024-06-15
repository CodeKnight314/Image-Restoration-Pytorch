import os 
from PIL import Image 
import argparse 
from glob import glob 
from tqdm import tqdm

def patch_generation(image_directory, output_directory, patch_width, patch_height, stride): 
    """
    Generates image patches from images in a directory and saves them to an output directory.

    Args:
        image_directory (str): The directory containing the original images.
        output_directory (str): The directory where the patches will be saved.
        patch_width (int): The width of each patch.
        patch_height (int): The height of each patch.
        stride (int): The number of pixels to move the patch window at each step.

    """
    if not os.path.exists(output_directory): 
        os.makedirs(output_directory)

    for filename in tqdm(os.listdir(image_directory)):
        if filename.lower().endswith((".png", ".jpg")): 
            img = Image.open(os.path.join(image_directory, filename)).convert("RGB")
            img_width, img_height = img.size

            patch_id = 0

            for i in range(0, img_width - patch_width, stride): 
                for j in range(0, img_height - patch_height, stride): 
                    box = (i, j, i + patch_width, j + patch_height)
                    
                    patch = img.crop(box=box)

                    patch_filename = f"{os.path.splitext(filename)[0]}_patch_{patch_id}.png"

                    patch.save(os.path.join(output_directory, patch_filename))
                    patch_id += 1

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="[HELP] Patch Generation given a directory of .png or .jpg.")
    parser.add_argument("--img_dir", type = str, help = "[HELP] Path to image directory")
    parser.add_argument("--out_dir", type = str, help = "[HELP] Path to output_directory")
    parser.add_argument("--patch_h", type = int, help = "[HELP] Patch Height")
    parser.add_argument("--patch_w", type = int, help = "[HELP] Patch Width")
    parser.add_argument("--stride", type = int, help = "[HELP] Stride per patch")

    args = parser.parse_args()

    patch_generation(args.img_dir, args.out_dir, args.patch_h, args.patch_w, args.stride)


        