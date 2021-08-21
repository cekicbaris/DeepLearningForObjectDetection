import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
from config import *
import json


ann_file = 'dataset/annotations/instances_val2017.json'
detections_file = 'eval/faster_rcnn.json'
#detections_file = 'eval/mask_rcnn.json'
#detections_file = 'eval/YOLOV3.json'
detections_file = 'eval/RetinaNet.json'

models = [
    {'model_name': 'Faster RCNN', 'filename':'faster_rcnn.json'},
    {'model_name': 'Mask RCNN', 'filename':'mask_rcnn.json'},
    {'model_name': 'RetinaNet', 'filename':'RetinaNet.json'},
    {'model_name': 'SSD', 'filename':'SSD.json'},
    {'model_name': 'YOLO v3', 'filename':'YOLOV3.json'},
    {'model_name': 'YOLO v5s', 'filename':'YOLOV5s.json'},
    {'model_name': 'YOLO v5x', 'filename':'YOLOV5x.json'},
    
]


ground_truth=COCO(ann_file)
imgIds=sorted(ground_truth.getImgIds())
#imgIds=imgIds[0:100]
#imgId = imgIds[np.random.randint(100)]

#coco_eval.params.catIds = [1] #person id : 1
#cocoEval.params.imgIds = imgIds   

#https://www.programmersought.com/article/3065285708/

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

stats =[]

for model in models:
    stat = {}
    print("ModelName : \t", model['model_name'] + "_________________________________________")
    detections_file = "eval/" + str(model['filename'])
    detections =ground_truth.loadRes(detections_file)
    cocoEval = COCOeval(ground_truth,detections,'bbox')
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    stat['model_name'] = model['model_name']
    stat['category'] = "all"
    for midx, measure in enumerate(measures):
        stat[measure] = cocoEval.stats[midx]
    stats.append(stat)

    for idx, category in enumerate(COCO_NAMES):
        if category != 'N/A' and category != '__background__':
            print("Category: " , category)
            cocoEval.params.catIds = [idx]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            stat = {}
            stat['model_name'] = model['model_name']
            stat['category'] = category
            for midx, measure in enumerate(measures):
                stat[measure] = cocoEval.stats[midx]
            stats.append(stat)


with open(MEASURES_FILE, 'w') as f:
    json.dump(stats , f)

data_file = open('data_file.csv', 'w')

print("finish")
