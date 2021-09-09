import torch
import numpy as np


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

COCO_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

IMG_INPUT_FOLDER = "dataset/input/"
IMG_OUTPUT_FOLDER = "dataset/output/"
EVALUATION_FOLDER = "eval/"
IMG_EXTENSION = ".jpg"
STAT_FILE = "stats/stats.json"
MEASURES_FILE = "stats/measures.json"
COCO_VALIDATION_SET_FILE = "dataset/annotations/instances_val2017.json"
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(COCO_NAMES), 3))


DEFAULT_IMAGES = [
                    'https://ultralytics.com/images/zidane.jpg',
                    'https://ultralytics.com/images/bus.jpg' 
                ]


# streamlit settings. 
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


MODELS = [
    {'model_id': 'faster_rcnn', 'model_name': 'Faster RCNN', 'filename':'faster_rcnn.json' },
    {'model_id': 'mask_rcnn', 'model_name': 'Mask RCNN', 'filename':'mask_rcnn.json'},
    {'model_id': 'retinanet', 'model_name': 'RetinaNet', 'filename':'RetinaNet.json'},
    {'model_id': 'ssd', 'model_name': 'SSD', 'filename':'SSD.json'},
    {'model_id': 'yolov3', 'model_name': 'YOLO v3', 'filename':'YOLOV3.json'},
    {'model_id': 'yolov5s', 'model_name': 'YOLO v5s', 'filename':'YOLOV5s.json'},
    {'model_id': 'yolov5x', 'model_name': 'YOLO v5x', 'filename':'YOLOV5x.json'},      
]

FASTERRCNN = MODELS[0]['model_id']
MASKRCNN = MODELS[1]['model_id']
RETINANET = MODELS[2]['model_id']
SINGLESHOTDETECTOR  = MODELS[3]['model_id']
YOLOV3 = MODELS[4]['model_id']
YOLOV5S = MODELS[5]['model_id']
YOLOV5X = MODELS[6]['model_id']
