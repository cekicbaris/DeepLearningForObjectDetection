import streamlit as st
import numpy as np
from config import * 
from experiment import * 
import cv2
from PIL import Image 
from torchvision import transforms as transforms


@st.cache
def process_image(uploaded_img):
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])
    
    image = Image.open(uploaded_img).convert('RGB')
    image = transform(image).to(DEVICE)

    faster_rcnn = FasterRCNN()
    faster_rcnn.measure_model_prediction([image],[0])

    processed_image = draw_boxes(faster_rcnn.boxes[0], faster_rcnn.labels[0], uploaded_img)
    return processed_image



st.title("Deep Learning for Object Detection")
uploaded_img = st.file_uploader("Please upload and image", type=["jpg", "jpeg"])
confidence_threshold = st.slider(
    "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
)

if uploaded_img is not None:
    processed_image = process_image(uploaded_img)
    st.image(
        processed_image, caption=f"Processed image", use_column_width=True,
    )
