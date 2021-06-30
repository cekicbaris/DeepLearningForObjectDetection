from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval 
import argparse
import numpy as np 
import skimage.io as io 
import pylab,json 

#--Gt: The path of the instances_val2014.json file under the coco dataset annotations path
# It should be noted that the prediction result format: a list consisting of a single bbox information dictionary:
# [
# {“bbox”: [225.7, 207.6, 128.7, 140.2], “score”: 0.999, “image_id”: 581929, “category_id”: 17}, 
# {“bbox”: [231.1, 110.6, 33.5, 36.7], “score”: 0.992, “image_id”: 581929, “category_id”: 17},
# …]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gt", type=str, help="Assign the groud true path.", default=None)
    parser.add_argument("-d", "--dt", type=str, help="Assign the detection result path.", default=None)
    args = parser.parse_args()
    
    cocoGt = COCO(args.gt)
    cocoDt = cocoGt.loadRes(args.dt)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()