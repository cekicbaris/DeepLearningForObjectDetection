from experiment import *
exp = Experiment(name="All_Models_Dry_Run", dry_run=True)


yolo_v5x = YOLO(version='V5X')
faster_rcnn = FasterRCNN()
yolo_v5s = YOLO(version='V5S')
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
exp.plot_results() # some models inference are higher than 5sec in CPU.