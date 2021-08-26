import time
from typing import List 
import torch
import torchvision
import numpy as np

import logging
from tqdm import tqdm
import torchvision.transforms as transforms

from dataclasses import dataclass
from torch.utils.data import DataLoader

import config
from toolkit import *
from dataset import * 




@dataclass
class Predictions:
    modelname:str
    boxes:list
    scores:list
    labels:list
    class_names:list
    stats:dict

@dataclass
class Evalutions:
    image_id:int
    filename:str
    category_id:int
    bbox:List
    score:float

class DetectionCompare():
    def __init__(self, images=config.DEFAULT_IMAGES ) -> None:
        
        # define the torchvision image transforms
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])

        self.modelname:str
        self.stats = {}
        self.evaluations = []
        self.xywh = []

    def predict(self, image):
        return self.predict_for_model(self.model, image)

    def predict_for_model(self,model, image, threshold=0.6):

        model.eval().to(config.DEVICE)
        with torch.no_grad():
            outputs = model(image) # get the predictions on the image
        

        scores = [outputs[i]['scores'].detach().cpu().numpy() for i in range(len(outputs))]
        score_mask = [scores[i] >= threshold for i in range(len(scores))]
        scores = [ scores[i][score_mask[i]] for i in range(len(scores))]

        bboxes = [outputs[i]['boxes'].detach().cpu().numpy() for i in range(len(outputs))]
        boxes = [ bboxes[i][score_mask[i]] for i in range(len(bboxes))]

        labels = [ outputs[i]['labels'].detach().cpu().numpy() for i in range(len(outputs))]
        labels = [ labels[i][score_mask[i]] for i in range(len(labels))]

        self.boxes, self.labels, self.scores = boxes, labels, scores

        
        #return self.results()
    
    def measure_model_prediction(self, imgs, coco_image_ids):
    # GPU measuring
    # https://deci.ai/resources/blog/measure-inference-time-deep-neural-networks/
        
    # CPU measuring    
        self.imgs = imgs
        self.coco_image_ids = coco_image_ids
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
        
        self.xywh = []
        for box in self.boxes:
            self.xywh.append(xyxy2xywh(box))       
        
        return self.results()
    
        
    def print_results(self):
        pass
        #              xmin        ymin         xmax        ymax  confidence  class    name
        # 0  749.628418   43.006378  1148.310181  708.739380    0.876501      0  person
        # 1  433.496307  433.949524   517.907959  715.133118    0.658130     27     tie
        # 2  113.315887  196.359955  1093.051270  710.308350    0.596343      0  person
        # 3  986.139587  304.344147  1027.974243  420.158539    0.285012     27     tie

    def results(self):
        self.class_names = [[ COCO_NAMES[j]   for j in self.labels[i] ] for i in range(len(self.labels))  ]
        predictions = Predictions(self.modelname, self.boxes, self.scores, self.labels, self.class_names, self.stats)
        self.results_toJSON = Predictions(self.modelname, [ a.tolist() for a in self.boxes], [s.tolist() for s in self.scores], [l.tolist() for l in self.labels], self.class_names, self.stats).__dict__

        for idx, coco_img_id in enumerate(self.coco_image_ids):
            for label_idx, label in enumerate(self.labels[idx].tolist()):
                evaluations = Evalutions(int(coco_img_id), coco_img_id,  label,self.xywh[idx].tolist()[label_idx], self.scores[idx].tolist()[label_idx]).__dict__
                self.evaluations.append(evaluations)
        return predictions
    

    def draw_box(self, img):
        _ , name = get_filename_from_path(img)
        filename = name + "_" + str(self.modelname) + config.IMG_EXTENSION
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
        self.modelname = MASKRCNN
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
            self.modelname = YOLOV5S
        elif self.version == 'V5x':
            model_to_load = 'ultralytics/yolov5'
            version_to_load = 'yolov5x'
            self.modelname = YOLOV5X
        else:
            model_to_load = 'ultralytics/yolov3'
            version_to_load = 'yolov3'
            self.modelname = YOLOV3
        
        # Model
        self.model = torch.hub.load(model_to_load, version_to_load, pretrained=True, )

    def predict(self, image):
        results = self.model(image)
        self.boxes = [results.xyxy[i][:,:4].detach().cpu().numpy() for i in range(len(results.xyxy))]
        self.scores = [results.xyxy[i][:,4].detach().cpu().numpy() for i in range(len(results.xyxy))]
        self.labels = [results.xyxy[i][:,5].detach().cpu().numpy().astype(np.int32) for i in range(len(results.xyxy))]
        self.labels = [ np.array([ COCO_NAMES.index(results.names[j]) for j in self.labels[i]])  for i in range(len(self.labels))  ]
        
        #return self.results()

if __name__ == "__main__":

    dataset = CustomDataset(resume=False)
    start_time = time.time()
    models = []
    if len(dataset.images) != 0:
        custom_images = DataLoader(dataset=dataset, batch_size=1)

        faster_rcnn = FasterRCNN()
        models.append(faster_rcnn)

        mask_rcnn = MaskRCNN()
        models.append(mask_rcnn)

        ssd = SSD()
        models.append(ssd)

        retinanet = RetinaNet()
        models.append(retinanet)

        yolo_v5x = YOLO(version='V5x')
        models.append(yolo_v5x)

        yolo_v5s = YOLO(version='V5s')
        models.append(yolo_v5s)

        yolo_v3 = YOLO(version='V3')
        models.append(yolo_v3)
               
        # if resume is enabled then first load the evaluations.
        if dataset.resume: 
            for model in models:
                dataset.read_eval(model.modelname)

        for idx, (imgs, gts, org_img) in tqdm(enumerate(custom_images)):

            _ , name = get_filename_from_path(org_img[0])
            coco_image_id = [name]
            image_stats ={}
            image_stats['original_image'] = org_img[0]
            image_stats['name'] = name
            
            for model in models:
                if 'yolo' in model.modelname :
                    model.measure_model_prediction(org_img[0], coco_image_id)    
                else:    
                    model.measure_model_prediction(imgs, coco_image_id)
                
                #model.measure_model_prediction(Image.open(org_img[0]), coco_image_id)                    
                
                #image_stats[model.modelname] = model.results_toJSON
                image_stats[model.modelname] = model.stats['duration']

                if idx % 2 == 0 and idx != 0:
                    model.draw_box(org_img[0])
                    dataset.save_stats() # checkpointing
                    dataset.save_eval(model.modelname, model.evaluations)
            
            dataset.add_stats(idx, org_img[0], image_stats)
            elapsed_time = ( time.time() - start_time )
            logging.info("---- Elapsed time : \t" + str(idx) + " - " + str(elapsed_time))
        
        #dataset.save_stats()

        for model in models:
            dataset.save_eval(model.modelname, model.evaluations)

        print("finished")
    else:
        logging.info("no images left for processing")

    elapsed_time = ( time.time() - start_time )
    logging.info("Total time : " + str(elapsed_time))