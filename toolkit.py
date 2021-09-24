import cv2
from config import *
from PIL import Image
import numpy as np
import ntpath
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pathlib import Path
import glob
import json
import config
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class Stats():
    def __init__(self,  experiment_folder='experiments/'):
        self.experiment_folder = experiment_folder 
        self.stats = []
        self.images = []
        self.measures = []
        self.stat_file = self.experiment_folder + config.STAT_FILE
    
    def add_stats(self, idx, original_img,  image_stats):
        self.stats.append(image_stats.copy())

    def check_path(self, file, create_if_not_exits=False):
        path = Path(file) 
        if path.exists():
            return True
        else:
            if create_if_not_exits:
                parent_directory_of_file = path.parent
                parent_directory_of_file.mkdir(parents=True, exist_ok=True)
                return True
            return False

    def save_stats(self):
        check_path(self.stat_file, True)
        with open(self.stat_file, 'w') as f:
            json.dump(self.stats , f)

    def read_stats(self, model_name):
        file = Path(self.stat_file)
        if file.exists():
            with open(self.stat_file, 'r') as f:
                stat_file = json.load(f)
        else:
            stat_file = []        
        return stat_file
    
    def save_eval(self, modelname, evaluations):
        filename = self.experiment_folder + config.EVALUATION_FOLDER + modelname + ".json"
        check_path(filename, True)
        with open(filename, 'w') as f:
            json.dump(evaluations , f)
    
    def save_summary(self, model_summary):
        filename = self.experiment_folder + config.MODEL_SUMMARY
        check_path(filename, True)
        with open(filename, 'w') as f:
            json.dump(model_summary , f)
    
    def summary_to_pandas(self):
        filename = self.experiment_folder + config.MODEL_SUMMARY
        with open(filename, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df

    def read_eval(self,modelname):
        filename = self.experiment_folder + config.EVALUATION_FOLDER + modelname + ".json"
        path = Path(filename) 
        if path.exists():
            with open(filename, 'r') as f:
                eval_file = json.load(f)
        else:
            eval_file = []        
        return eval_file

    def save_measures(self):
        filename = self.experiment_folder + config.MEASURES_FILE 
        check_path(filename, True)
        with open(filename, 'w') as f:
            json.dump(self.measures , f)

    def list_processed_files(self):
        stat_file = self.read_stats()
        processed_files = [i['original_image'] for i in stat_file]
        return processed_files
    
    def list_unprocess_files(self):
        processed = self.list_processed_files()
        unprocessed = [x for x in self.images if x not in processed]
        return unprocessed


def check_path(file, create_if_not_exits=False):
    path = Path(file) 
    if path.exists():
        return True
    else:
        if create_if_not_exits:
            parent_directory_of_file = path.parent
            parent_directory_of_file.mkdir(parents=True, exist_ok=True)
            return True
        return False

 

def describe_stats(stat_files):
    stat_dfs = [pd.read_json(file) for file in stat_files]
    final_df = pd.concat(stat_dfs, axis=1)
    describe_num_df = final_df.describe().drop(columns=['name'])
    describe_num_df.reset_index(inplace=True)
    return describe_num_df


def draw_boxes(boxes, classes, image, save=False, filename=None):
    image = cv2.cvtColor(np.array(Image.open(image).convert('RGB')), cv2.COLOR_RGB2BGR)
    
    for i, box in enumerate(boxes):
        #color = COLORS[COCO_NAMES.index(classes[i])]
        color = COLORS[classes[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, str(COCO_NAMES[classes[i]]), (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, 
                    lineType=cv2.LINE_4)
    if save:
         cv2.imwrite(f"{filename}", image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     
    return image


def get_filename_from_path(path,return_only_name=False):
    filename = ntpath.basename(path)
    k = filename.rfind(".")
    if return_only_name:
        return filename[:k]
    return filename, filename[:k]

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    #y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    #y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def ground_truth_bbox(image_id):
    coco = COCO(COCO_VALIDATION_SET_FILE)
    coco_all_cats_idx = coco.getCatIds()
    coco_categories_names = coco.loadCats(coco_all_cats_idx)


    annotation_ids = coco.getAnnIds(imgIds=[int(image_id)], 
                        catIds=coco_all_cats_idx, iscrowd=None)
    image_annotations = coco.loadAnns(annotation_ids)
    image_categories = []
    image_boxes = []

    image = IMG_INPUT_FOLDER + 'val2017/' + image_id + IMG_EXTENSION

    for annotation in image_annotations:
        image_categories.append(annotation['category_id'])
        bbox = annotation['bbox']
        # x, y, width, height => ( coordinates ) x, y, x + width , y + height
        image_boxes.append([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])

    filename =  image_id + "_GT" +  IMG_EXTENSION
    draw_boxes(image_boxes, image_categories, image, save=True, filename=filename)


def evaluate(model_name, detections_file=None, by_category=False, image_ids=[]):
    #https://www.programmersought.com/article/3065285708/
    ann_file = COCO_VALIDATION_SET_FILE

    if image_ids == []:
        ground_truth=COCO(ann_file)
        image_ids=sorted(ground_truth.getImgIds())

    if detections_file == None:
        detections_file = "eval/" + model_name + '.json'
        
    ground_truth=COCO(ann_file)
    imgIds=image_ids

    measures = [
                "AP",
                "AP05",
                "AP075",
                "AP_small",
                "AP_medium",
                "AP_large",
                "AR",
                "AR05",
                "AR075",
                "AR_small",
                "AR_medium",
                "AR_large"
                ]

    stats = []
    stat = {}
    detections =ground_truth.loadRes(detections_file)
    cocoEval = COCOeval(ground_truth,detections,'bbox')
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    stat['model_name'] = model_name
    stat['category'] = "all"
    for midx, measure in enumerate(measures):
        stat[measure] = cocoEval.stats[midx]    
    stats.append(stat)

    if by_category:
        for idx, category in enumerate(COCO_NAMES):
            if category != 'N/A' and category != '__background__':
                #print("Category: " , category)
                cocoEval.params.catIds = [idx]
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                stat = {}
                stat['model_name'] = model_name
                stat['category'] = category
                for midx, measure in enumerate(measures):
                    stat[measure] = cocoEval.stats[midx]
                stats.append(stat)

    return stats


    


if __name__ == "__main__":
    #ground_truth_bbox("000000020333")

    """ evaluate([int('000000007108')],"mask_rcnn", by_category=False)
    evaluate([int('000000007108')],"faster_rcnn", by_category=False)
    evaluate([int('000000007108')],"SSD", by_category=False)
    evaluate([int('000000007108')],"RetinaNet", by_category=False)
    evaluate([int('000000007108')],"YOLOV3", by_category=False)
    evaluate([int('000000007108')],"YOLOV5s", by_category=False)
    evaluate([int('000000007108')],"YOLOV5x", by_category=False)
     """
    


