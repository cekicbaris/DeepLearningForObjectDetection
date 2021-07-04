import config
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image 
from pathlib import Path
import glob


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms



class CustomDataset(Dataset):
    def __init__(self, images=[]):
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])
        files = glob.glob(str( Path(config.IMG_INPUT_FOLDER) / '**' / '*.*'), recursive=True)
        self.images = sorted([x.replace('/', os.sep) for x in files if x.split('.')[-1].lower() in config.IMG_FORMATS])
        sa = os.sep + 'images' + os.sep
        sb = os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
        self.label_files =  ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in self.images]


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        ground_truth = []

        #data_folder = Path("input/")
        #original_image = str(data_folder.absolute() / self.images[index])
        original_image = self.images[index]
        image = Image.open(original_image).convert('RGB')

        image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        image = self.transform(image).to(config.DEVICE)

        #image = image.unsqueeze(0) # add a batch dimension
        return image, ground_truth, original_image
