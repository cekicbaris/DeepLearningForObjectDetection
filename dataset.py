import config
import numpy as np
import os
import pandas as pd
import torch
from PIL import Image 
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, images=[]):
        self.images = [
                    'input/zidane.jpg',
                    'input/bus.jpg' 
                ]
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        ground_truth = []

        #data_folder = Path("input/")
        #original_image = str(data_folder.absolute() / self.images[index])
        original_image = self.images[index]
        image = Image.open(original_image).convert('RGB')
        image = self.transform(image).to(config.DEVICE)
        #image = image.unsqueeze(0) # add a batch dimension
        return image, ground_truth, original_image
