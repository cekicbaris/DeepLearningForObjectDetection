import cv2
from config import COCO_NAMES, COLORS, IMG_OUTPUT_FOLDER
from PIL import Image
import numpy as np
import ntpath
import torch


def draw_boxes(boxes, classes, image, save=False, filename=None):
    image = cv2.cvtColor(np.array(Image.open(image).convert('RGB')), cv2.COLOR_RGB2BGR)
    
    for i, box in enumerate(boxes):
        #color = COLORS[COCO_NAMES.index(classes[i])]
        color = COLORS[classes[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, str(COCO_NAMES[classes[i]]), (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    if save:
         cv2.imwrite(f"{IMG_OUTPUT_FOLDER}{filename}", image)
    return image


def get_filename_from_path(path):
    filename = ntpath.basename(path)
    k = filename.rfind(".")
    return filename, filename[:k]

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    #y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    #y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

if __name__ == "__main__":
    pass

