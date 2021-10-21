
import io
from config import * 
from experiment import * 

from fastapi import FastAPI, File, UploadFile

from PIL import Image
import json


MODELS = [
        {'model_name': 'yolov5s', 'model' : YOLO(version='V5s') },
        {'model_name': 'yolov5x', 'model' : YOLO(version='V5x') },
        {'model_name': 'yolov3', 'model' : YOLO(version='V3') },
        {'model_name': 'faster_rcnn', 'model' : FasterRCNN() },
        {'model_name': 'mask_rcnn', 'model' : MaskRCNN() },
        {'model_name': 'ssd', 'model' : SSD() },
        {'model_name': 'retinanet', 'model' : RetinaNet() },    
        ]


app = FastAPI(
    title = "Deep Learning for Object Detection",
    copyright="Mehmet Baris Cekic, City, University of London",
    version = "0.1.0"
)

async def object_detection(algorithm, file_bytes):
     #name = f"/data/{str(uuid.uuid4())}.png"

    for model in MODELS : 
        if algorithm == model['model_name']:
            selected_model = model
            break
            
    print(selected_model)
    result = {}
    if selected_model:
        if 'yolo' in selected_model['model_name']:
            uploaded_img = Image.open(io.BytesIO(file_bytes))
            selected_model['model'].measure_model_prediction(uploaded_img,[0])
        else:
            transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])
            uploaded_img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            uploaded_img = transform(uploaded_img).to(DEVICE)
    
            selected_model['model'].measure_model_prediction([uploaded_img],[0])
        #processed_image = draw_boxes(yolo_v5s.boxes[0], yolo_v5s.labels[0], uploaded_img)
        result['boxes'] = selected_model['model'].boxes[0].tolist() 
        result['scores'] = selected_model['model'].scores[0].tolist() 
        result['labes'] =  [ COCO_NAMES[label] for label in selected_model['model'].labels[0].tolist()]
    
    #result = json.dumps(result)
    return result

@app.post("/detect", summary="Object Detection by Different Algorithms", tags=['Generic'] )
async def detect(algorithm, file: UploadFile = File(...)):
    file_bytes = file.file.read()
    return await object_detection(algorithm, file_bytes)

@app.post("/yolov5s", summary="Object Detection by YOLO V5s", tags=['YOLO Family'])
async def yolov5s(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    return await object_detection('yolov5s', file_bytes)

@app.post("/yolov5x", summary="Object Detection by YOLO V5x", tags=['YOLO Family'])
async def yolov5x(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    return await object_detection('yolov5x', file_bytes)

@app.post("/yolov3", summary="Object Detection by YOLO 3", tags=['YOLO Family'])
async def yolov3(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    return await object_detection('yolov3', file_bytes)

@app.post("/fasterrcnn", summary="Object Detection by Faster R-CNN", tags=['R-CNN Family'])
async def fasterrcnn(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    return await object_detection('faster_rcnn', file_bytes)

@app.post("/maskrcnn", summary="Object Detection by Mask R-CNN", tags=['R-CNN Family'])
async def maskrcnn(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    return await object_detection('mask_rcnn', file_bytes)
    
@app.post("/retinanet", summary="Object Detection by RetinaNet", tags=['Other'])
async def retinanet(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    return await object_detection('retinanet', file_bytes)

@app.post("/SSD", summary="Object Detection by SSD", tags=['Other'])
async def retinanet(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    return await object_detection('SSD', file_bytes)