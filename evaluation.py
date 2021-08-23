import json

from config import *
from toolkit import * 


ann_file = COCO_VALIDATION_SET_FILE

models = [
    {'model_name': 'Faster RCNN', 'filename':'faster_rcnn.json'},
    {'model_name': 'Mask RCNN', 'filename':'mask_rcnn.json'},
    {'model_name': 'RetinaNet', 'filename':'RetinaNet.json'},
    {'model_name': 'SSD', 'filename':'SSD.json'},
    {'model_name': 'YOLO v3', 'filename':'YOLOV3.json'},
    {'model_name': 'YOLO v5s', 'filename':'YOLOV5s.json'},
    {'model_name': 'YOLO v5x', 'filename':'YOLOV5x.json'}    
]

ground_truth=COCO(ann_file)
imgIds=sorted(ground_truth.getImgIds())

for model in models:
    print("ModelName : \t", model['model_name'] + "_________________________________________")
    stats = evaluate(imgIds, model['model_name'], detections_file = model['filename'])

with open(MEASURES_FILE, 'w') as f:
    json.dump(stats , f)

print("finish")
