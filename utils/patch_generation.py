import os
from PIL import Image
import argparse
from tqdm import tqdm

def patch_generation(image_directory: str, output_directory: str, patch_width: int, patch_height: int, stride: int):
    """
    Generates image patches from images in a directory and saves them to an output directory,
    preserving the subdirectory structure.

    Args:
        image_directory (str): The directory containing the original images.
        output_directory (str): The directory where the patches will be saved.
        patch_width (int): The width of each patch.
        patch_height (int): The height of each patch.
        stride (int): The number of pixels to move the patch window at each step.
    """
    for root, _, files in os.walk(image_directory):
        for filename in tqdm(files):
            if filename.lower().endswith((".png", ".jpg")):
                img_path = os.path.join(root, filename)
                img = Image.open(img_path).convert("RGB")
                img_width, img_height = img.size

                relative_path = os.path.relpath(root, image_directory)
                output_subdir = os.path.join(output_directory, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                patch_id = 0
                for i in range(0, img_width - patch_width + 1, stride):
                    for j in range(0, img_height - patch_height + 1, stride):
                        box = (i, j, i + patch_width, j + patch_height)
                        patch = img.crop(box=box)
                        patch_filename = f"{os.path.splitext(filename)[0]}_patch_{patch_id}.png"
                        patch.save(os.path.join(output_subdir, patch_filename))
                        patch_id += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="[HELP] Patch Generation given a directory of .png or .jpg.")
    parser.add_argument("--img_dir", type=str, required=True, help="[HELP] Path to image directory")
    parser.add_argument("--out_dir", type=str, required=True, help="[HELP] Path to output directory")
    parser.add_argument("--patch_h", type=int, required=True, help="[HELP] Patch Height")
    parser.add_argument("--patch_w", type=int, required=True, help="[HELP] Patch Width")
    parser.add_argument("--stride", type=int, required=True, help="[HELP] Stride per patch")

    args = parser.parse_args()

    patch_generation(args.img_dir, args.out_dir, args.patch_w, args.patch_h, args.stride)