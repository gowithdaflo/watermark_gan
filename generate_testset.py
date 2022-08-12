from watermark_gan.model.model import Generator, Discriminator
from watermark_gan.training.data_pipeline import WatermarkDataset, Scale
from watermark_gan.utils.utils import plot_tensors, transfrom_tensor_for_plot
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import random

np.random.seed(42343)
random.seed(42343)

if __name__ == '__main__':
    
    workdir = "/".join(__file__.split("/")[:2])
    size = 256
    dataset = WatermarkDataset(f"{workdir}/gans/datasets/watermarks/train/no-watermark",
                            transform=transforms.Compose([Scale(size)]))
    testset = {"inputs": [], "targets": [], "mask": []}
    for i in range(10,):        
        sample = dataset[i]
        for k, v in sample.items():
            testset[k].append(v)
    testset = {k: np.stack(v, axis=0) for k, v in testset.items()}
    
    np.savez(f"{workdir}/gans/datasets/watermarks/test.npz", **testset)
    
    
    # testset = np.load(f"{workdir}/gans/datasets/watermarks/test.npz")
    
    # testset = {k: v for k, v in testset.items()}
    
    # fig, axs = plt.subplots(10,3, figsize=(5,15))
    # for i in range(10,):
    #     inputs = torch.FloatTensor(testset["inputs"][i])
    #     targets = torch.FloatTensor(testset["targets"][i])
    #     mask = torch.FloatTensor(testset["mask"][i])
    #     axs[i,0].imshow(transfrom_tensor_for_plot(inputs))
    #     axs[i,1].imshow(transfrom_tensor_for_plot(targets))
    #     axs[i,2].imshow(transfrom_tensor_for_plot(mask*2-1))

    # plt.show()
