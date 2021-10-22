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
    """
    Data class to store prediction results. 
    """
    modelname:str
    boxes:list
    scores:list
    labels:list
    class_names:list
    stats:dict

@dataclass
class Evalutions:
    """
    Data class to store evaluation results. 
    """
    image_id:int
    filename:str
    category_id:int
    bbox:List
    score:float

class Detection():
    """
    This is the main detection class, it will serve as base class to models. 
    Every object detection algorihm implements this base class.
    """
    def __init__(self, images=config.DEFAULT_IMAGES ) -> None:
        self.modelname:str
        self.stats = {}
        self.evaluations = []
        self.xywh = []

    def predict(self, image):
        """
        This proxy method is responsible to make prediction.  
        It will can be overriden.
        """
        return self.predict_for_model(image,self.model,)

    def predict_for_model(self,image,model, threshold=0.6):
        """
        This method is responsible to make prediction.  

        Args:
            image ([tensor]): tensor
            model ([string]): detection algorithm
            threshold (float, optional): Defaults to 0.6.
        """

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
        """     
            # model prediction performance on CPU and GPU
            # for GPU performance
            # https://deci.ai/resources/blog/measure-inference-time-deep-neural-networks/
       
        Args:
            imgs ([type]): [description]
            coco_image_ids ([type]): [description]

        """
 
        # CPU measuring    
        self.imgs = imgs
        self.coco_image_ids = coco_image_ids
        if torch.cuda.is_available():
            # GPU measurement is differnet than CPU , therefore torch utility is used istead of time.time()
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
            duration = (time.time() - s ) * 1000
        
        logging.info(self.modelname + ":\t\t" + str(duration))
        self.stats['duration'] = duration
        
        self.xywh = []
        for box in self.boxes:
            self.xywh.append(xyxy2xywh(box))       
        
        return self.results()

    def results(self):
        """
            This method is reponsible to prepare results in prediction data class and json.
        Returns:
            prediction as data class
        """
        self.class_names = [[ COCO_NAMES[j]   for j in self.labels[i] ] for i in range(len(self.labels))  ]
        predictions = Predictions(self.modelname, self.boxes, self.scores, self.labels, self.class_names, self.stats)
        self.results_toJSON = Predictions(self.modelname, [ a.tolist() for a in self.boxes], [s.tolist() for s in self.scores], [l.tolist() for l in self.labels], self.class_names, self.stats).__dict__

        for idx, coco_img_id in enumerate(self.coco_image_ids):
            for label_idx, label in enumerate(self.labels[idx].tolist()):
                evaluations = Evalutions(int(coco_img_id), coco_img_id,  label,self.xywh[idx].tolist()[label_idx], self.scores[idx].tolist()[label_idx]).__dict__
                self.evaluations.append(evaluations)
        return predictions
    
    def draw_box(self, img, experiment_name):
        """
        draws bounding box to the predicted image.  
        """
        _ , name = get_filename_from_path(img)
        filename = config.IMG_OUTPUT_FOLDER + experiment_name + '/' + name + "_" + str(self.modelname) + config.IMG_EXTENSION
        check_path(filename, True)
        draw_boxes(self.boxes[0], self.labels[0], img ,  save=True, filename=filename)


class FasterRCNN(Detection):
    """
    Faster RCNN implementation of Detection Experiment Framework
    https://pytorch.org/vision/stable/models.html#id57
    """
    def __init__(self, ):
        super().__init__()
        self.modelname = FASTERRCNN
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

class MaskRCNN(Detection):
    """
    Mask RCNN implementation of Detection Experiment Framework
    https://pytorch.org/vision/stable/models.html#id63

    """
    def __init__(self, ):
        super().__init__()
        self.modelname = MASK_RCNN
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

class RetinaNet(Detection):
    """
    RetinaNet implementation of Detection Experiment Framework
    https://pytorch.org/vision/stable/models.html#id58

    """
    
    def __init__(self, min_size=800):
        super().__init__()
        self.modelname = RETINA_NET
        self.min_size = min_size
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True, min_size=self.min_size)

class SSD(Detection):
    """
    SSD implementation of of Detection Experiment Framework
    https://pytorch.org/vision/stable/models.html#id59
    """
    def __init__(self):
        super().__init__()
        self.modelname = SINGLESHOTDETECTOR
        self.model = torchvision.models.detection.ssd300_vgg16(pretrained=True)

class YOLO(Detection):
    """
    YOLO implementation of of Detection Experiment Framework
    https://pytorch.org/hub/ultralytics_yolov5/
    """   
    def __init__(self, version='V3', model_name_only=False):
        super().__init__()
        self.version = version.lower()
        if self.version == 'v5s':
            model_to_load = 'ultralytics/yolov5'
            version_to_load = YOLOV5S
            self.modelname = YOLOV5S
        elif self.version == 'v5x':
            model_to_load = 'ultralytics/yolov5'
            version_to_load = YOLOV5X
            self.modelname = YOLOV5X
        else:
            model_to_load = 'ultralytics/yolov3'
            version_to_load = YOLOV3
            self.modelname = YOLOV3
        
        # Model
        if model_name_only:
            self.model = ""
        else:
            self.model = torch.hub.load(model_to_load, version_to_load, pretrained=True, )

    def predict(self, image):
        """
        YOLO spesific prediction impelementation, 
        PytorchHub model is different then the torchvision implementation, therefore it is overriden. 
        """
        results = self.model(image)
        self.boxes = [results.xyxy[i][:,:4].detach().cpu().numpy() for i in range(len(results.xyxy))]
        self.scores = [results.xyxy[i][:,4].detach().cpu().numpy() for i in range(len(results.xyxy))]
        self.labels = [results.xyxy[i][:,5].detach().cpu().numpy().astype(np.int32) for i in range(len(results.xyxy))]
        self.labels = [ np.array([ COCO_NAMES.index(results.names[j]) for j in self.labels[i]])  for i in range(len(self.labels))  ]


class Experiment():
    """
    This class is the experiment class.
    It adds models to experiment scopes, collect stats and store every 
    different experiment(name) to differnt folders in experiments folder
    """
    def __init__(self, name, resume=False, dry_run=True) -> None:
        self.experiment_name = name
        
        self.experiment_folder =  config.EXPERIMENTS_FOLDER + self.experiment_name + '/'
        self.experiment_state_file = self.experiment_folder + config.EXPERIMENT_STATE_FILE

        self.resume = False

        self.models = []
        self.stats_collector = Stats(experiment_folder=self.experiment_folder)
        self.dry_run = dry_run
        
        self.images, self.stats = [],[]

        self.load_dataset()

    def add_model(self,model):
        self.models.append(model)

    def load_dataset(self):
        self.dataset = ExperimentDataset(images=self.images, resume=self.resume, dry_run=self.dry_run)
        self.images = DataLoader(dataset=self.dataset, batch_size=1)

        if len(self.dataset.images) == 0:
            logging.warning('no image in dataset')

    def run_experiment(self):
        """
        This method iterates over dataloader for every image and runs 
        the prediction in the models directory of experiment. 
        """
        start_time = time.time()

        for idx, (imgs, gts, org_img) in enumerate(self.images):
            name = get_filename_from_path(org_img[0], True)
            coco_image_id = [name]
            
            image_stats ={}
            image_stats['original_image'] = org_img[0]
            image_stats['name'] = name
            
            for model in self.models:
                if 'yolo' in model.modelname :
                    model.measure_model_prediction(org_img[0], coco_image_id)    
                else:    
                    model.measure_model_prediction(imgs, coco_image_id)

                image_stats[model.modelname] = model.stats['duration']

                if idx % 5 == 0 and idx != 0:
                    model.draw_box(org_img[0], experiment_name = self.experiment_name)
                    self.stats_collector.save_stats() # checkpointing
                    self.stats_collector.save_eval(model.modelname, model.evaluations)
            
            self.stats_collector.add_stats(idx, org_img[0], image_stats)
            elapsed_time = ( time.time() - start_time )
            logging.info("---- Elapsed time : \t" + str(idx) + " - " + str(elapsed_time))
        
        self.stats_collector.save_stats()

        for model in self.models:
            self.stats_collector.save_eval(model.modelname, model.evaluations)
    
    def evaluate_results(self, previous_result_to_merge=None, by_category=True):
        """
        This method is responsible to generate mAP, mAR for MSCOCO style IoU thresholds and  inference type in ms. 
        """
        stats = []
        summary = []
        
        for model in self.models:
            print("ModelName : \t", model.modelname + "_________________________________________")
            stat = evaluate(model.modelname, detections_file = self.experiment_folder + 'eval/' +  model.modelname + '.json',by_category=by_category, image_ids=self.dataset.image_ids )
            model_summary = {}
            model_summary['model_name'] = model.modelname
            model_summary['AP'] = stat[0]['AP']
            summary.append(model_summary)
            stats.append(stat)
        
        self.stats_collector.measures = stats
        self.stats_collector.save_measures()

        
        stat_files = [self.stats_collector.stat_file]
        if previous_result_to_merge:
            stat_files = [self.stats_collector.stat_file,previous_result_to_merge]
        stat_dfs = [pd.read_json(file) for file in stat_files]
        final_df = pd.concat(stat_dfs, axis=1)
        describe_num_df = final_df.describe().drop(columns=['name'])
        describe_num_df.reset_index(inplace=True)

        for idx, model_summary in enumerate(summary):
            summary[idx]['Inference'] = describe_num_df[model_summary['model_name']][1]

        self.stats_collector.save_summary(summary)
        return self.stats_collector.summary_to_pandas()
    
    def plot_results(self, scale=500, mAP=1):
        df = self.stats_collector.summary_to_pandas()
        if df.empty:
            df = self.evaluate_results()
        
        sns.relplot(x="Inference", y="AP", data=df, hue="model_name", s=500)
        sns.set(style="ticks")
        plt.grid()
        plt.xlabel('Inference Speed in ms ( '+ self.experiment_name +')')
        plt.ylabel('mAP')
        plt.xlim(0, scale)
        plt.ylim(0, mAP)

        # add annotations one by one with a loop
        for line in range(0,df.shape[0]):
            plt.text(df.Inference[line]-0.10, df.AP[line]+0.02, df.model_name[line], horizontalalignment='left', size='medium', color='black', weight='regular')

        filename = self.experiment_folder + config.MODEL_SUMMARY_PLOT
        plt.savefig(filename)
        print("Result file :", filename)
        print("Experiment folder :",  self.experiment_folder)


if __name__ == "__main__":
    """
    Unite tests
    """
    exp = Experiment(name="Unit_Test", dry_run=True)
    ssd = SSD()
    yolov5x = YOLO(version='v5x')
    exp.add_model(yolov5x)
    exp.add_model(ssd)
    #exp.run_experiment()
    exp.evaluate_results()
    exp.plot_results()