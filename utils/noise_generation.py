import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import glob

def generate_gaussian_noise(mean: float, std: float, size: tuple) -> np.ndarray:
    """
    Generates Gaussian noise.

    Args:
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
        size (tuple): Size of the noise array.

    Returns:
        np.ndarray: The generated Gaussian noise.
    """
    return np.random.normal(mean, std, size)

def preprocess_images(input_dir: str, output_dir: str, levels: list[float]):
    """
    Adds Gaussian noise to images in the input directory at different noise levels and saves them in the output directory.

    Args:
        input_dir (str): Directory containing the original images.
        output_dir (str): Directory to save the noised images.
        levels (list[float]): List of noise levels (standard deviations) to apply.

    """
    os.makedirs(output_dir, exist_ok=True)

    for level in levels:
        level_dir = os.path.join(output_dir, f"level_{level}")
        os.makedirs(level_dir, exist_ok=True)

        # Gather all image files (png, jpg, jpeg) recursively
        image_files = glob.glob(os.path.join(input_dir, '**', '*.png'), recursive=True) + \
                      glob.glob(os.path.join(input_dir, '**', '*.jpg'), recursive=True) + \
                      glob.glob(os.path.join(input_dir, '**', '*.jpeg'), recursive=True)

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
    """
    Main function to parse arguments and preprocess images.

    Args:
        args: Parsed command-line arguments.
    """
    input_dir = args.input_dir
    output_dir = args.output_dir
    levels = args.levels

    preprocess_images(input_dir, output_dir, levels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gaussian noise at different levels and preprocess images.")
    parser.add_argument("input_dir", help="Input directory containing the folder of sharp images")
    parser.add_argument("output_dir", help="Output directory to store the preprocessed images")
    parser.add_argument("--levels", nargs="+", type=float, default=[15, 25, 50], help="Levels of noise standard deviation to apply (default: [15, 25, 50])")

    args = parser.parse_args()
    main(args)
