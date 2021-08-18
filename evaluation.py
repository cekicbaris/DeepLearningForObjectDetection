import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io

ann_file = 'dataset/annotations/instances_val2017.json'
ann_file = 'dataset/annotations/val_139.json'
ground_truth=COCO(ann_file)


#detections_file = 'eval/YOLOV5s.json'

detections_file = 'eval/eval_test.json'
detections =ground_truth.loadRes(detections_file)

imgIds=sorted(ground_truth.getImgIds())
#imgIds=imgIds[0:100]
#imgId = imgIds[np.random.randint(100)]

cocoEval = COCOeval(ground_truth,detections,'bbox')
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

print("finish")
