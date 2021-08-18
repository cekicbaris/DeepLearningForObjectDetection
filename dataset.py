import config
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image 
from pathlib import Path
import glob
import json


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms



class CustomDataset(Dataset):
    def __init__(self, images=[], resume=False):
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])
        files = glob.glob(str( Path(config.IMG_INPUT_FOLDER) / '**' / '*.*'), recursive=True)
        self.images = sorted([x.replace('/', os.sep) for x in files if x.split('.')[-1].lower() in config.IMG_FORMATS])
        self.stats = []
        self.resume = resume
        if self.resume:
            self.images = self.list_unprocess_files()
            self.stats = self.read_stats()

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

    def add_stats(self, idx, original_img,  image_stats):
        self.stats.append(image_stats.copy())
        
    def save_stats(self):
        with open(config.STAT_FILE, 'w') as f:
            json.dump(self.stats , f)

    def read_stats(self):
        file = Path(config.STAT_FILE)
        if file.exists():
            with open(config.STAT_FILE, 'r') as f:
                stat_file = json.load(f)
        else:
            stat_file = []        
        return stat_file


    def save_eval(self, modelname, evaluations):
        filename = config.EVALUATION_FOLDER + modelname + ".json"
        with open(filename, 'w') as f:
            json.dump(evaluations , f)

    def read_eval(self,modelname):
        filename = config.EVALUATION_FOLDER + modelname + ".json"
        path = Path(filename) 
        if path.exists():
            with open(filename, 'r') as f:
                eval_file = json.load(f)
        else:
            eval_file = []        
        return eval_file


    def list_processed_files(self):
        stat_file = self.read_stats()
        processed_files = [i['original_image'] for i in stat_file]
        return processed_files
    
    def list_unprocess_files(self):
        processed = self.list_processed_files()
        unprocessed = [x for x in self.images if x not in processed]
        return unprocessed

