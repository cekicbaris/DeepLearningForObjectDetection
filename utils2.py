import cv2
from config import COCO_NAMES, COLORS

def draw_boxes(boxes, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[COCO_NAMES.index(classes[i])]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image

def save_image(image, filename, path):
    cv2.imwrite(f"outputs/{filename}.jpg", image)
