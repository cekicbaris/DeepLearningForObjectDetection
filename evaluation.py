import json

from config import *
from toolkit import * 


ann_file = COCO_VALIDATION_SET_FILE

ground_truth=COCO(ann_file)
imgIds=sorted(ground_truth.getImgIds())

for model in MODELS:
    print("ModelName : \t", model['model_name'] + "_________________________________________")
    stats = evaluate(imgIds, model['model_name'], detections_file = model['filename'])

with open(MEASURES_FILE, 'w') as f:
    json.dump(stats , f)

print("finish")
