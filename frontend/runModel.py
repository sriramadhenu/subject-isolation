import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modelCode.Unet import UNET
from modelCode.Fcn import FCN
from PIL import ImageOps

def predict_and_visualize(model_name, image_path, device, transforms_func, num_classes):
    
    if model_name == "UNET":
        model = UNET(n_channels=3, n_classes=num_classes)
        model.load_state_dict(torch.load('../models/UNET/unet_voc_final.pth', map_location=device))
        model.to(device)
    elif model_name == "FCN":
        model = FCN(n_classes=num_classes).to(device)
        model.load_state_dict(torch.load('../models/FCN/fcn_voc_final.pth', map_location=device))
        model.to(device)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
    model.eval()
   
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    original_size = image.size[::-1] # H, W

    # Use only the image transform part for prediction
    input_tensor = transforms_func.transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # Get the prediction (class with highest score)
    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Resize mask back to original image size
    pred_mask_pil = Image.fromarray(pred_mask.astype(np.uint8)).resize(original_size, Image.NEAREST)

    # Create a colormap for visualization (PASCAL VOC standard)
    def create_pascal_label_colormap():
        colormap = np.zeros((256, 3), dtype=int)
        ind = np.arange(256, dtype=int)
        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3
        return colormap

    colormap = create_pascal_label_colormap()
    colored_mask = colormap[pred_mask_pil]
    original_size = np.array(image).shape[:2][::-1]  # (width, height)

    # Resize mask to match original image size
    colored_mask_resized = Image.fromarray(colored_mask.astype(np.uint8)).resize(original_size)
    colored_mask = np.array(colored_mask_resized)
       # Overlay the mask onto the original image
    overlay_image = image.resize(original_size)
    mask_rgba = Image.fromarray(colored_mask.astype(np.uint8)).convert("RGBA")
    overlay_image_rgba = overlay_image.convert("RGBA")

    # Blend with transparency
    blended = Image.blend(overlay_image_rgba, mask_rgba, alpha=0.5)

    # Display
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(np.array(image))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(colored_mask.astype(np.uint8))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(np.array(blended))
    plt.axis('off')

    plt.tight_layout()
    os.makedirs('static', exist_ok=True)
    plt.savefig("static/prediction_output.png")
    plt.close()




