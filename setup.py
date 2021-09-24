import os

os.system("pip3 install -r requirements.txt")
os.system("mkdir tmp")

coco_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
coco_annocations_local_file = 'tmp/annotations_trainval2017.zip'

coco_val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
coco_val_images_local_file = 'tmp/val2017.zip'

# COCO Annotations
os.system(f"curl -L '{coco_annotations_url}' -o '{coco_annocations_local_file}' --retry 3 -C -")
os.system(f"unzip -o '{coco_annocations_local_file}' -d ./dataset/")

# COCO Validation Images
os.system(f"curl -L '{coco_val_images_url}' -o '{coco_val_images_local_file}' --retry 3 -C -")
os.system(f"unzip -o '{coco_val_images_local_file}' -d ./dataset/input/")

os.system("rm -r tmp*")
