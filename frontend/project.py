from flask import Flask, render_template, request, url_for
from PIL import Image
import numpy as np
import torch
from modelCode.Unet import VOCTransforms
from modelCode.runModel import predict_and_visualize
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        image = request.files['file']
        selected_model = request.form.get('Model')
        # Device setup
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        # Fallback to CPU
        else:
            device = torch.device('cpu')
        num_classes = 21
        IMG_SIZE = (256, 256)
        transforms_func = voc_transforms = VOCTransforms(IMG_SIZE)
        print(f"Using device: {device}")
        predict_and_visualize(selected_model, image, device, transforms_func, num_classes)
        return render_template('index.html', prediction_url=url_for('static', filename='prediction_output.png'), disabled="disabled")

if __name__ == '__main__':
    app.run()