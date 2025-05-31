from flask import Flask, render_template, request
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')

    image = request.files['file']
    img = Image.open(image)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run()