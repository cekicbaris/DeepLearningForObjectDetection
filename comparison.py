from sys import version
from dataclasses import dataclass
import torch
import torchvision
import numpy as np
from torchvision.models.detection import retinanet
from torchvision.models.detection.retinanet import RetinaNet
import torchvision.transforms as transforms
import cv2
from PIL import Image
import logging

import config
from toolkit import *
from dataset import * 


RETINANET = "RetinaNet"
SINGLESHOTDETECTOR  = "SSD(Single Shot Detector)"
FASTERRCNN = "Faster RCNN"
YOLOV3 = "YOLO v3"
YOLOV5 = "YOLO v5"

@dataclass
class Predictions:
    modelname:str
    boxes:list
    scores:list
    labels:list


class DetectionCompare():
    def __init__(self, images=config.DEFAULT_IMAGES ) -> None:
        
        # define the torchvision image transforms
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])

        self.modelname:str

    def predict(self, image):
        return self.predict_for_model(self.model, image)

    def predict_for_model(self,model, image, threshold=0.6):

        model.eval().to(config.DEVICE)
        with torch.no_grad():
            outputs = model(image) # get the predictions on the image
        # get all the scores
        scores = outputs[0]['scores'].detach().cpu().numpy()

        score_mask = scores >= threshold

        # get all the predicted bounding boxes
        # get boxes above the threshold score
        bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        boxes = bboxes[score_mask]
        labels = outputs[0]['labels'].cpu().numpy()
        labels = labels[score_mask].astype(np.int32)
        scores = scores[score_mask]
        self.boxes, self.labels, self.scores = boxes, labels, scores
        return self.results()

    def print_results(self):
        pass
        #              xmin        ymin         xmax        ymax  confidence  class    name
        # 0  749.628418   43.006378  1148.310181  708.739380    0.876501      0  person
        # 1  433.496307  433.949524   517.907959  715.133118    0.658130     27     tie
        # 2  113.315887  196.359955  1093.051270  710.308350    0.596343      0  person
        # 3  986.139587  304.344147  1027.974243  420.158539    0.285012     27     tie

    # def draw_boxes(self, boxes, classes):
    #     for idx, img in enumerate(self.images):
    #         draw_boxes(boxes[idx], classes[idx], img)

    def results(self):
        predictions = Predictions(self.modelname, self.boxes, self.scores, self.labels)
        return predictions

    # def __convert_classes_to_labels(self, classes):
    #     #pred_classes = [coco_names[labels[i]] for i in thresholded_preds_inidices]
    #     pass

class FasterRCNN(DetectionCompare):
    def __init__(self, ):
        super().__init__()
        self.modelname = FASTERRCNN
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


class RetinaNet(DetectionCompare):
    def __init__(self, min_size=800):
        super().__init__()
        self.modelname = RETINANET
        self.min_size = min_size
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, min_size=self.min_size)

class SSD(DetectionCompare):
    def __init__(self):
        super().__init__()
        self.modelname = SINGLESHOTDETECTOR
        self.model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

class YOLO(DetectionCompare):
    def __init__(self, version='V3'):
        super().__init__()
        self.version = version
        self.modelname = 'YOLO' + str(version)
        if self.version == 'V5':
            model_to_load = 'ultralytics/yolov5'
            version_to_load = 'yolov5s'
        else:
            model_to_load = 'ultralytics/yolov3'
            version_to_load = 'yolov3'
        
        # Model
        self.model = torch.hub.load(model_to_load, version_to_load, pretrained=True)

    def predict(self, image):
        results = self.model(image)
        #results.save()
        #TODO : Update with boxes, classes, scores
        return None, None, None


def measure_model_prediction(model:DetectionCompare, imgs):
# GPU measuring
# https://deci.ai/resources/blog/measure-inference-time-deep-neural-networks/
    
# CPU measuring    
    if torch.cuda.is_available():
      starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
      starter.record()
    else:
      s = time.time()

    results = model.predict(imgs)
    if torch.cuda.is_available():
      #print("cuda measurement")
      ender.record()
      torch.cuda.synchronize()
      duration = starter.elapsed_time(ender)
    else:
      duration = (time.time() - s ) * 1000
    
    print(model.modelname, duration)
    return results, duration

if __name__ == "__main__":
    import timeit
    from torch.utils.data import DataLoader
    import time 
    dataset = CustomDataset()
    custom_images = DataLoader(dataset=dataset, batch_size=1)

    faster_rcnn = FasterRCNN()
    ssd = SSD()
    retinanet = RetinaNet()
    yolo = YOLO(version='V5')



    for idx, (imgs, gts, org_img) in enumerate(custom_images):
        faster_rcnn_results, _ = measure_model_prediction(faster_rcnn, imgs) #faster_rcnn.predict(imgs)
        ssd_results, _ = measure_model_prediction(ssd, imgs)  #ssd.predict(imgs)
        retinanet_results, _ = measure_model_prediction(retinanet, imgs)  # retinanet.predict(imgs)
        yolo_results, _ = measure_model_prediction(yolo, org_img[0])  #yolo.predict(org_img[0])
        print("finished")
    
    

    


