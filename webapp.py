import streamlit as st
import numpy as np
from config import * 
from experiment import * 
from PIL import Image 
from torchvision import transforms as transforms

@st.cache
def process_image(uploaded_img, algorithm):
    transform = transforms.Compose([
                            transforms.ToTensor(),
                            ])
    
    image = Image.open(uploaded_img).convert('RGB')
    image = transform(image).to(DEVICE)

    model = [eval(model['model_definition']) for model in MODELS if model['model_name'] == algorithm]
    model = model[0]

    #model.measure_model_prediction(Image.open(uploaded_img),[0])
    start_time = time.time()
    if 'yolo' in algorithm.lower():
        model.measure_model_prediction(Image.open(uploaded_img),[0])
    else:
        model.measure_model_prediction([image],[0])  

    end_time = time.time()

    processed_image = draw_boxes(model.boxes[0], model.labels[0], uploaded_img)
    process_duration = ( end_time - start_time ) * 1000
    return processed_image, process_duration

st.title("Deep Learning for Object Detection")
st.sidebar.title("Settings")

alg = {}

algorithm = st.sidebar.radio("Select Detection Algorithm",
        [model['model_name'] for model in MODELS])

uploaded_img = st.file_uploader("Please upload and image", type=["jpg", "jpeg"])

st.subheader(algorithm)

if uploaded_img is not None:
    processed_image, duration = process_image(uploaded_img, algorithm)
    st.code('Inference Time : '+ str(duration) + ' miliseconds')
    st.image(
        processed_image, caption=f"Processed image", use_column_width=True,
    )

