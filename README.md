# Deep Learning for Object Detection
**Mehmet Baris Cekic**, 2021 MSc Artificial Intelligence, City, University of London

The experiment is implemented as a PyTorch model which loads each image and use selected models to detect objects in each image by predicting bounding box(es) and then evaluate results by calculating experiment measures detailed in section 0 All selected models are pretrained on MSCOCO Dataset, and MSCOCO validation set which consists of 5000 images, will be used for this experiment. All selected models are given MSCOCO 2017 validation set images and predict bounding boxes for each image and store inference times (ms). Later, evaluation measures such as precision, recall, mean average precision(mAP) for different IoU thresh-old and image scales (small, medium, large), are calculated. 

The implementation has three main classes. 
*  `DetectionModels(experiment.py)`  class is a base class and re-sponsible for prediction and measurement. All model specific models implement DetectionMod-els class. YOLO has a version property and overwrites predict method. One can extend the ex-periment scope with new models by implementing DetectionModels class. 
*  `ExperimentDataset(dataset.py)` class implements Dataset from torch.utils.data and it is responsible to load images, transform it to tensors.  
*  `Experiment(exp_prodK80_all.py)` class is the main class to load detection models, datasets and run the experiments and generate results and store it for review. 


Setup 
* `pip3 install -r requirements.txt`
* `python3 setup.py`

Run and Existing Experiment
* `python3 exp_prodK80_all.py`

Conduct a new experiment
```

from experiment import *

exp = Experiment(name="Custom Experiment Name Goes Here", dry_run=False)

yolo_v5s = YOLO(version='V5S')
yolo_v5x = YOLO(version='V5X')
faster_rcnn = FasterRCNN()
mask_rcnn = MaskRCNN()
retinanet = RetinaNet()
ssd = SSD()

exp.add_model(yolo_v5s)
exp.add_model(yolo_v5x) 
exp.add_model(mask_rcnn)
exp.add_model(faster_rcnn)
exp.add_model(retinanet)
exp.add_model(ssd)


exp.run_experiment()
exp.evaluate_results()

exp.evaluate_results()
exp.plot_results(scale=6000) # some models inference are higher than 5sec in CPU.

```


To run Object Detection Web App
`streamlit run app.py`

To Run Object Detection Rest API
`uvicorn restAPI:app --reload`

