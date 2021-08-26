import streamlit as st
import numpy as np
from config import * 
from experiment import * 
from PIL import Image 
from torchvision import transforms as transforms

@st.cache
def process_image(uploaded_img):
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])
    
    image = Image.open(uploaded_img).convert('RGB')
    image = transform(image).to(DEVICE)

    #faster_rcnn = FasterRCNN()
    #faster_rcnn.measure_model_prediction([image],[0])

    yolo_v5s = YOLO(version='V5s')
    yolo_v5s.measure_model_prediction(Image.open(uploaded_img),[0])

    processed_image = draw_boxes(yolo_v5s.boxes[0], yolo_v5s.labels[0], uploaded_img)
    return processed_image

st.title("Deep Learning for Object Detection")
st.sidebar.title("Settings")

alg = {}

algorithm = st.sidebar.radio("Select Detection Algorithm",
        [model['model_name'] for model in MODELS])

uploaded_img = st.file_uploader("Please upload and image", type=["jpg", "jpeg"])

st.subheader(algorithm)

if uploaded_img is not None:
    processed_image = process_image(uploaded_img)
    st.image(
        processed_image, caption=f"Processed image", use_column_width=True,
    )
