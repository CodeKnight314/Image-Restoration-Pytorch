import cv2 
import numpy as np
from typing import Tuple, Union
from glob import glob 
from tqdm import tqdm 
import os
import matplotlib.pyplot as plt
from collections import Counter
import torch
from PIL import Image

def add_gaussian_noise(image_path : str, mean : int, std : int, output_directory : Union[str, None], show : bool = False): 
    """
    Adds Gaussian Noise to a given image with a specified mean and standardeviation 

    Args: 
        image_path (str): direct directory to the image 
        mean (int): mean distribution of gaussian noise 
        std (int): standard deviation of the guassian noise distribution
        output_directory (Union[str, None]): If directory is specified, image will be saved to specified directory
        show (bool): Shows image and destroys window after pressing key
    """ 
    image = cv2.imread(image_path)

    gaussian_noise = np.zeros(image.shape, dtype = np.uint8)
    
    cv2.randn(gaussian_noise, mean=mean, std=std)
    
    gaussian_noise = (gaussian_noise * 0.5).astype(np.uint8)
    
    image = cv2.add(image, gaussian_noise)
    
    if show: 
        cv2.imshow(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_directory:
        cv2.imwrite(output_directory,image)

    return image

def add_uniform_noise(image_path : str, output_directory : Union[str, None], lower_bound : int, upper_bound : int, show : bool = False):
    """
    Adds Uniform Noise to a given image with a specified lower and upper bound. 

    Args: 
        image_path (str): direct directory to the image 
        lower_bound (int): lower bound of the uniform distribution
        upper_bound (int): upper bound of the uniform distribution
        output_directory (Union[str, None]): If directory is specified, image will be saved to specified directory
        show (bool): Shows image and destroys window after pressing key
    """
    image = cv2.imread(image_path)

    uni_noise = np.zeros(image.shape, dtype = np.unint8)

    cv2.randu(uni_noise, low=lower_bound, high=upper_bound)

    image = cv2.add(image, uni_noise)

    if show: 
            cv2.imshow(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if output_directory:
        cv2.imwrite(output_directory,image)

    return image

def add_impulse_noise(image_path : str, output_directory : Union[str, None], lower_bound : int, upper_bound : int, show : bool = False):
    """
    Adds Impulse Noise (Pepper Noise) to a given image.
    
    Args: 
        image_path (str): direct directory to the image 
        lower_bound (int): lower bound of the uniform distribution
        upper_bound (int): upper bound of the uniform distribution
        output_directory (Union[str, None]): If directory is specified, image will be saved to specified directory
        show (bool): Shows image and destroys window after pressing key

    """
    image = cv2.imread(image_path)

    imp_noise = np.zeros(image.shape, dtype = np.unint8)

    cv2.randu(imp_noise, low=lower_bound, high=upper_bound)
    
    imp_noise = cv2.threshold(imp_noise,245,255,cv2.THRESH_BINARY)[1]

    image = cv2.add(image, imp_noise)

    if show: 
            cv2.imshow(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if output_directory:
        cv2.imwrite(output_directory,image)

    return image

def batch_noise(root_dir : str, output_dir : Union[str, None], show : bool = False, mode : str = "gaussian", **kwargs):
    """
    Applies noise to all images in a specified directory and saves the modified images to an output directory. 
    The function supports different types of noise such as Gaussian, uniform, and impulse.

    Args:
        root_dir (str): The directory containing the images to process.
        output_dir (Union[str, None]): The directory where the noised images will be saved. If None, images are not saved.
        show (bool): If True, displays the noised image. Defaults to False.
        mode (str): The type of noise to apply. Options include 'gaussian', 'uniform', or 'impulse'.
        **kwargs: Keyword arguments specific to the type of noise:
            For 'gaussian':
                mean (float): The mean of the Gaussian noise.
                std (float): The standard deviation of the Gaussian noise.
            For 'uniform' and 'impulse':
                lower_bound (float): The lower bound of the noise distribution.
                upper_bound (float): The upper bound of the noise distribution.

    Raises:
        ValueError: If an invalid mode is specified.
    """
    image_paths = glob(os.path.join(root_dir, "/*"))
    if mode.lower() == "gaussian":           
        for image in tqdm(image_paths): 
            add_gaussian_noise(image_path=image, 
                               output_directory=os.path.join(output_dir, os.path.basename(image).split("/")[-1]), 
                               mean = kwargs['mean'], std=kwargs['std'], 
                               show=False)
    elif mode.lower() == "uniform": 
        for image in tqdm(image_paths): 
            add_uniform_noise(image_path=image, 
                              output_directory=os.path.join(output_dir, os.path.basename(image).split("/")[-1]),
                              lower_bound=kwargs['lower_bound'], upper_bound=kwargs['upper_bound'], 
                              show=False)
    elif mode.lower() == "impulse":
        for image in tqdm(image_paths): 
            add_uniform_noise(image_path=image, 
                              output_directory=os.path.join(output_dir, os.path.basename(image).split("/")[-1]),
                              lower_bound=kwargs['lower_bound'], upper_bound=kwargs['upper_bound'], 
                              show=False)
    else: 
        raise ValueError(f"[Error] Invalid mode. {mode} is not available as a noise mode.")

def count_labels(directory):
    """
    Counts the number of images in each label directory within the given parent directory.
    
    Args:
    directory (str): The path to the directory containing labeled subdirectories of images.
    
    Returns:
    dict: A dictionary with keys as labels and values as counts of images.
    """
    label_counts = Counter()
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            label_counts[label] = len([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])
    
    return dict(label_counts)

def plot_data(output_log, output_directory, save_fig, show_fig): 
    """
    Plots training and validation loss and accuracy over epochs from a log file.

    Args:
        output_log (str): The path to the log file containing the training data.
        output_directory (str): The directory where the plots will be saved if `save_fig` is True.
        save_fig (bool): Whether to save the plots to the output directory.
        show_fig (bool): Whether to display the plots.

    """
    with open(output_log, 'r') as file:
        lines = file.readlines()

    tr_loss_ls = []
    val_loss_ls = []
    accuracy_ls = []
     
    for line in lines:
        parts = line.split(" ")
        if len(parts) > 12:  # Ensure the line has enough parts
            tr_loss_ls.append(float(parts[4]))
            val_loss_ls.append(float(parts[8]))
            accuracy_ls.append(float(parts[12]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(tr_loss_ls) + 1), tr_loss_ls, label="Training Loss")
    plt.plot(range(1, len(val_loss_ls) + 1), val_loss_ls, label="Validation Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend() 

    if save_fig: 
        plt.savefig(os.path.join(output_directory, "Training_Validation_Loss.png"))

    if show_fig: 
        plt.show() 
    
    plt.close() 

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracy_ls) + 1), accuracy_ls, label="Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend() 

    if save_fig: 
        plt.savefig(os.path.join(output_directory, "Accuracy.png"))
    
    if show_fig: 
        plt.show() 

    plt.close()

def tensor_to_pil_save(tensor, path):
    """
    Converts an RGB Tensor to a PIL image and saves it.

    Args:
        tensor (torch.Tensor): The input tensor to be converted. Expected shape is (C, H, W).
        path (str): The file path where the image will be saved.
    """
    tensor = tensor.to('cpu').clone()
    
    tensor = tensor.squeeze(0)  
    tensor = torch.clamp(tensor, 0, 1)  
    tensor = tensor.mul(255).byte()  
    
    tensor = tensor.permute(1, 2, 0).numpy()  
    pil_image = Image.fromarray(tensor)
    
    pil_image.save(path)