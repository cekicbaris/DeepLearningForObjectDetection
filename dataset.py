import config
import numpy as np
import os
from PIL import Image 
from pathlib import Path
import glob
from toolkit import * 

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, images=[], resume=False, dry_run=False):
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize(
                            #       mean=[0.485, 0.457, 0.407],
                            #       std=[1,1,1])
                            ])

        
        files = glob.glob(str( Path(config.IMG_INPUT_FOLDER) / '**' / '*.*'), recursive=True)
        if dry_run:
            files = glob.glob(str( Path(config.IMG_INPUT_FOLDER_DRY_RUN) / '**' / '*.*'), recursive=True)
        self.images = sorted([x.replace('/', os.sep) for x in files if x.split('.')[-1].lower() in config.IMG_FORMATS])

        self.resume = resume
        if self.resume:
            self.images = images


        sa = os.sep + 'images' + os.sep
        sb = os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
        self.label_files =  ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in self.images]

        self.image_ids = [ int(get_filename_from_path(x, return_only_name = True)) for x in self.images]
        

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        ground_truth = []
        original_image = self.images[index]
        image = Image.open(original_image).convert('RGB')
        image = self.transform(image).to(config.DEVICE)
        return image, ground_truth, original_image

