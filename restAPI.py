
import io
from fastapi import FastAPI, File, UploadFile
#from fastapi.responses import HTMLResponse

from config import * 
from experiment import * 
from PIL import Image
import json


app = FastAPI(
    title = "Deep Learning for Object Detection",
    copyright="Mehmet Baris Cekic, City, University of London",
    version = "0.1.0"
)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    uploaded_img = Image.open(io.BytesIO(file_bytes))
    #name = f"/data/{str(uuid.uuid4())}.png"

    # image.save(name)
    #image.filename = name
    
    yolo_v5s = YOLO(version='V5s')
    yolo_v5s.measure_model_prediction(uploaded_img,[0])

    #processed_image = draw_boxes(yolo_v5s.boxes[0], yolo_v5s.labels[0], uploaded_img)
    result = {}
    result['boxes'] = yolo_v5s.boxes[0].tolist() 
    result['labes'] = yolo_v5s.labels[0].tolist() 
    
    #result = json.dumps(result)
    return result
    