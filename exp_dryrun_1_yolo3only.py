from numpy import mod
from experiment import *

exp = Experiment(name="All_Models_Dry_Run", dry_run=True)

yolo_v3 = YOLO(version='V3')
exp.add_model(yolo_v3)
exp.run_experiment()

exp.evaluate_results()
exp.plot_results() # some models inference are higher than 5sec in CPU.