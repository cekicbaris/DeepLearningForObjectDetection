import torch
import numpy as np

# PyTorch device variable to run the model on CPU or on GPU(cuda)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MSCOCO 91 classes
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

# Stat collector settings
IMG_INPUT_FOLDER = "dataset/input/val2017/"
IMG_INPUT_FOLDER_DRY_RUN = "dataset/input_small/"
IMG_OUTPUT_FOLDER = "dataset/output/"
EVALUATION_FOLDER = "eval/"
IMG_EXTENSION = ".jpg"
STAT_FILE = "stats/stats.json"
MEASURES_FILE = "stats/measures.json"
COCO_VALIDATION_SET_FILE = "dataset/annotations/instances_val2017.json"
EXPERIMENTS_FOLDER  = 'experiments/' 
EXPERIMENT_STATE_FILE = 'experiment.json'
MODEL_SUMMARY = 'stats/modelsummary.json'
MODEL_SUMMARY_PLOT = 'stats/modelsummary.png'

#Allowed Image file extenions. 
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(COCO_NAMES), 3))

# To run model with default images. 
DEFAULT_IMAGES = [
                    'https://ultralytics.com/images/zidane.jpg',
                    'https://ultralytics.com/images/bus.jpg' 
                ]

# streamlit settings. 
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


YOLOV5S = 'yolov5s'
YOLOV3 = 'yolov3'
YOLOV5X = 'yolov5x'
FASTERRCNN = 'faster_rcnn'
MASK_RCNN = 'mask_rcnn'
RETINA_NET = 'retinanet'
SINGLESHOTDETECTOR = 'SSD'

MODELS = [
    {'model_id': YOLOV5S, 'model_name': 'YOLO v5s', 'filename':'YOLOV5s.json','model_definition' : 'YOLO(version="V5s")'},
    {'model_id': YOLOV3 , 'model_name': 'YOLO v3', 'filename':'YOLOV3.json','model_definition' : 'YOLO(version="V3")'},
    {'model_id': YOLOV5X, 'model_name': 'YOLO v5x', 'filename':'YOLOV5x.json','model_definition' : 'YOLO(version="V5x")'},      
    {'model_id': FASTERRCNN, 'model_name': 'Faster RCNN', 'filename':'faster_rcnn.json' , 'model_definition' : 'FasterRCNN()'},
    {'model_id': MASK_RCNN, 'model_name': 'Mask RCNN', 'filename':'mask_rcnn.json' , 'model_definition' : 'MaskRCNN()'},
    {'model_id': RETINA_NET, 'model_name': 'RetinaNet', 'filename':'RetinaNet.json','model_definition' : 'RetinaNet()'},
    {'model_id': SINGLESHOTDETECTOR, 'model_name': 'SSD', 'filename':'SSD.json', 'model_definition' : 'RetinaNet()'},
   
]

