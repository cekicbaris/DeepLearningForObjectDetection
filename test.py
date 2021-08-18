import json


with open("dataset/annotations/instances_val2017.json") as jsonFile:
    anno = json.load(jsonFile)

newfile = {}
new_anno = []
new_images = []

newfile['info'] = anno['info']



for anx in anno['annotations']:
    if anx['image_id'] == 139:
        annotation = {}
        for key in anx.keys():
            if key != 'segmentation':
                annotation[key] = anx[key]
        new_anno.append(annotation)

for anx in anno['images']:
    if anx['id'] == 139:
        new_images.append(anx)

newfile['annotations'] = new_anno
newfile['images'] = new_images
newfile['categories'] = anno['categories']


with open('dataset/annotations/val_139.json', 'w') as jsonFile:
    json.dump(newfile, jsonFile)


print("loaded")
