import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import glob

def generate_gaussian_noise(mean, std, size):
    return np.random.normal(mean, std, size)

def preprocess_images(input_dir, output_dir, levels):
    os.makedirs(output_dir, exist_ok=True)

    for level in levels:
        level_dir = os.path.join(output_dir, f"level_{level}")
        os.makedirs(level_dir, exist_ok=True)

        image_files = glob.glob(os.path.join(input_dir, '**', '*.png'), recursive=True) + glob.glob(os.path.join(input_dir, '**', '*.jpg'), recursive=True) + glob.glob(os.path.join(input_dir, '**', '*.jpeg'), recursive=True)

        for image_path in tqdm(image_files, desc=f"Processing level {level}"):
            if not os.path.basename(image_path).startswith("._"):
                image = cv2.imread(image_path)

                if image is not None:
                    noise = generate_gaussian_noise(0, level, image.shape)
                    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)

                    rel_image_path = os.path.relpath(image_path, input_dir)
                    output_path = os.path.join(level_dir, rel_image_path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, noisy_image)

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    levels = args.levels

    preprocess_images(input_dir, output_dir, levels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gaussian noise at different levels and preprocess images.")
    parser.add_argument("input_dir", help="Input directory containing the folder of sharp images")
    parser.add_argument("output_dir", help="Output directory to store the preprocessed images")
    parser.add_argument("--levels", nargs="+", type=float, default=[15,25, 50], help="Levels of noise standard deviation to apply (default: [15, 25, 50])")

    args = parser.parse_args()
    main(args)
