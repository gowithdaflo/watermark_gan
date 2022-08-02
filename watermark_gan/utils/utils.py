import matplotlib.pyplot as plt
import numpy as np
import glob
import torch

def uint8_to_float(array):
    return array/127.5 - 1

def float_to_uint8(tensor):
    return ((tensor + 1) * 127.5).type(torch.uint8)

def transfrom_tensor_for_plot(tensor):
    return float_to_uint8(tensor.permute(1,2,0))

def plot_sample(sample):
    fig, axs = plt.subplots(1,3)

    axs[0].imshow(transfrom_tensor_for_plot(sample["inputs"]))
    axs[1].imshow(transfrom_tensor_for_plot(sample["targets"]))
    axs[2].imshow(sample["mask"].permute(1,2,0))
    plt.show()
    
def plot_tensors(tensor_list: list):
    fig, axs = plt.subplots(1, len(tensor_list))

    for i, tensor in enumerate(tensor_list):
        axs[i].imshow(transfrom_tensor_for_plot(tensor))
    plt.show()
    
def add_result_to_plot(generator, sample, ax):
    image = sample["inputs"]
    image = image.unsqueeze(0)
    out, mask = generator(image)
    
    mask = torch.nn.Sigmoid()(mask)*2-1
    target_mask = sample["mask"]*2-1
    ax[0].imshow(image)
    ax[1].imshow(out)
    ax[2].imshow(mask)
    ax[3].imshow(target_mask)
    

def find_files(dir_path, pattern, recursive=True):
    """
    Find all files that match a pattern in a directory (and subdirectories).
    
    Args:
        dir_path (str): Path of the directory.

        pattern (str): Pattern to match.

        recursive (bool): If True also look in subdirectories.

    Returns:
        list: List of found files that match the pattern.
    """
    deliminator = "/**/" if recursive else "/"
    files = glob.glob(f"{dir_path}{deliminator}{pattern}", recursive=recursive)
    return files