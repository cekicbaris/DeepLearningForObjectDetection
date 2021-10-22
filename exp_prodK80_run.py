
from numpy import mod
from experiment import *

exp = Experiment(name="Prod Run-K80", dry_run=False)

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

# hacky solution to overcome library name duplicate in Pytorch hub model
yolo_v3 = YOLO(version='V3', model_name_only=True)
exp.add_model(yolo_v3)
exp.evaluate_results(previous_result_to_merge= "experiments/" + str(exp.experiment_name) + "/stats/stats_yolov3.json")
exp.plot_results() # some models inference are higher than 5sec in CPU.