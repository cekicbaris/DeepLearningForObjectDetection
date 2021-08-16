import time 
import torch
import torchvision
import numpy as np

import torchvision.transforms as transforms
import cv2
import logging
import ntpath

from sys import version
from dataclasses import dataclass

from torchvision.models.detection import retinanet
from torchvision.models.detection.retinanet import RetinaNet
from torch.utils.data import DataLoader
from PIL import Image


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
    class_names:list
    stats:dict

class DetectionCompare():
    def __init__(self, images=config.DEFAULT_IMAGES ) -> None:
        
        # define the torchvision image transforms
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])

        self.modelname:str
        self.stats = {}

    def predict(self, image):
        return self.predict_for_model(self.model, image)


    def predict_for_model(self,model, image, threshold=0.6):

        model.eval().to(config.DEVICE)
        with torch.no_grad():
            outputs = model(image) # get the predictions on the image
        # get all the scores
        #scores = outputs[0]['scores'].detach().cpu().numpy()
        #score_mask = scores >= threshold
        # get all the predicted bounding boxes
        # get boxes above the threshold score
        #bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        #boxes = bboxes[score_mask]
        #labels = outputs[0]['labels'].cpu().numpy()
        #labels = labels[score_mask].astype(np.int32)
        #scores = scores[score_mask]
        

        scores = [outputs[i]['scores'].detach().cpu().numpy() for i in range(len(outputs))]
        score_mask = [scores[i] >= threshold for i in range(len(scores))]
        scores = [ scores[i][score_mask[i]] for i in range(len(scores))]

        bboxes = [outputs[i]['boxes'].detach().cpu().numpy() for i in range(len(outputs))]
        boxes = [ bboxes[i][score_mask[i]] for i in range(len(bboxes))]

        labels = [ outputs[i]['labels'].detach().cpu().numpy() for i in range(len(outputs))]
        labels = [ labels[i][score_mask[i]] for i in range(len(labels))]

        
        
        
        self.boxes, self.labels, self.scores = boxes, labels, scores
        #return self.results()
    
    def measure_model_prediction(self, imgs):
    # GPU measuring
    # https://deci.ai/resources/blog/measure-inference-time-deep-neural-networks/
        
    # CPU measuring    
        self.imgs = imgs
        if torch.cuda.is_available():
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
        else:
            s = time.time()

        self.predict(imgs)
        
        if torch.cuda.is_available():
        #print("cuda measurement")
            ender.record()
            torch.cuda.synchronize()
            duration = starter.elapsed_time(ender)
        else:
            duration = (time.time() - s ) 
        
        logging.info(self.modelname + ":\t\t" + str(duration))
        self.stats['duration'] = duration
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
        self.class_names = [[ COCO_NAMES[j]   for j in self.labels[i] ] for i in range(len(self.labels))  ]
        predictions = Predictions(self.modelname, self.boxes, self.scores, self.labels, self.class_names, self.stats)
        self.results_toJSON = Predictions(self.modelname, [ a.tolist() for a in self.boxes], [s.tolist() for s in self.scores], [l.tolist() for l in self.labels], self.class_names, self.stats).__dict__
        return predictions
    

    def draw_box(self, img):
        _ , name = get_filename_from_path(img)
        filename = name + "_" + str(self.modelname)
        draw_boxes(self.boxes[0], self.labels[0], img ,  save=True, filename=filename)
    
    # def __convert_classes_to_labels(self, classes):
    #     #pred_classes = [coco_names[labels[i]] for i in thresholded_preds_inidices]
    #     pass

class FasterRCNN(DetectionCompare):
    def __init__(self, ):
        super().__init__()
        self.modelname = FASTERRCNN
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

class MaskRCNN(DetectionCompare):
    def __init__(self, ):
        super().__init__()
        self.modelname = FASTERRCNN
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

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
        if self.version == 'V5s':
            model_to_load = 'ultralytics/yolov5'
            version_to_load = 'yolov5s'
        elif self.version == 'V5x':
            model_to_load = 'ultralytics/yolov5'
            version_to_load = 'yolov5x'
        else:
            model_to_load = 'ultralytics/yolov3'
            version_to_load = 'yolov3'
        
        # Model
        self.modelname = 'YOLO' + str(version)
        self.model = torch.hub.load(model_to_load, version_to_load, pretrained=True, )

    def predict(self, image):
        results = self.model(image)
        self.boxes = [results.xyxy[i][:,:4].detach().cpu().numpy() for i in range(len(results.xyxy))]
        self.scores = [results.xyxy[i][:,4].detach().cpu().numpy() for i in range(len(results.xyxy))]
        self.labels = [results.xyxy[i][:,5].detach().cpu().numpy().astype(np.int32) for i in range(len(results.xyxy))]
        self.labels = [ np.array([ COCO_NAMES.index(results.names[j]) for j in self.labels[i]])  for i in range(len(self.labels))  ]
        
        return self.results()



if __name__ == "__main__":


    dataset = CustomDataset(resume=True)
    start_time = time.time()
    if len(dataset.images) != 0:
        custom_images = DataLoader(dataset=dataset, batch_size=1)

        faster_rcnn = FasterRCNN()
        mask_rcnn = MaskRCNN()
        ssd = SSD()
        retinanet = RetinaNet()
        yolo_v5x = YOLO(version='V5x')
        yolo_v5s = YOLO(version='V5s')
        yolo_v3 = YOLO(version='V3')
        #yolo = YOLO(version='V5x')

        image_stats ={}

        for idx, (imgs, gts, org_img ) in enumerate(custom_images):

            faster_rcnn_results = faster_rcnn.measure_model_prediction(imgs) #faster_rcnn.predict(imgs)
            mask_rcnn_results = mask_rcnn.measure_model_prediction(imgs) #faster_rcnn.predict(imgs)
            ssd_results = ssd.measure_model_prediction( imgs)  #ssd.predict(imgs)
            retinanet_results= retinanet.measure_model_prediction( imgs)  # retinanet.predict(imgs)
            yolo_v5x_results = yolo_v5x.measure_model_prediction(org_img[0])  #yolo.predict(org_img[0])
            yolo_v5s_results = yolo_v5s.measure_model_prediction(org_img[0])  #yolo.predict(org_img[0])
            yolo_v3_results = yolo_v3.measure_model_prediction(org_img[0])  #yolo.predict(org_img[0])

            if idx % 50 == 0:
                faster_rcnn.draw_box(org_img[0])
                ssd.draw_box(org_img[0])
                retinanet.draw_box(org_img[0])
                yolo_v3.draw_box(org_img[0])
                yolo_v5s.draw_box(org_img[0])
                yolo_v5x.draw_box(org_img[0])
                dataset.save_stats() # checkpointing

            _ , name = get_filename_from_path(org_img[0])
            image_stats['original_image'] = org_img[0]
            image_stats['name'] = name
            image_stats['faster_rcnn_results'] = faster_rcnn.results_toJSON
            image_stats['mask_rcnn_results'] = mask_rcnn.results_toJSON
            image_stats['ssd_results'] = ssd.results_toJSON
            image_stats['retinanet_results'] = retinanet.results_toJSON
            image_stats['yolo_v5x_results'] = yolo_v5x.results_toJSON
            image_stats['yolo_v5s_results'] = yolo_v5s.results_toJSON
            image_stats['yolo_v3_results'] = yolo_v3.results_toJSON

            dataset.add_stats(idx, org_img[0], image_stats)
            elapsed_time = ( time.time() - start_time )
            logging.info("---- Elapsed time : \t" + str(idx) + " - " + str(elapsed_time))
        dataset.save_stats()
        print("finished")
    else:
        logging.info("no images left for processing")

    elapsed_time = ( time.time() - start_time )
    logging.info("Total time : " + str(elapsed_time))

    

    


