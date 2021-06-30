import torch

# Model
model = torch.hub.load('ultralytics/yolov3', 'yolov3')  # or 'yolov3_spp', 'yolov3_tiny'

# Image
img = 'https://ultralytics.com/images/zidane.jpg'

# Inference
results = model(img)
results.print()  # or .show(), .save()
results.save()