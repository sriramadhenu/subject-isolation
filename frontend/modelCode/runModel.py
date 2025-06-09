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
from .Unet import UNET
from .Fcn import FCN
from PIL import ImageOps
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from torchvision.transforms import PILToTensor, ToTensor

def predict_and_visualize(model_name, image_path, device, transforms_func, num_classes):
    
    if model_name == "UNET":
        model = UNET(n_channels=3, n_classes=num_classes)
        model.load_state_dict(torch.load('../models/UNET/unet_voc_final.pth', map_location=device))
        model.to(device)
    elif model_name == "FCN":
        model = FCN(n_classes=num_classes).to(device)
        model.load_state_dict(torch.load('../models/FCN/fcn_voc_final.pth', map_location=device))
        model.to(device)
    elif model_name=="K-Means":
        predict_kmeans_mask(image_path)
        return
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


def predict_kmeans_mask(image_path):
    performance_size = (256, 256)
    image = Image.open(image_path).convert("RGB")
    image.thumbnail(performance_size, Image.Resampling.LANCZOS)
    image_tensor = ToTensor()(image)
    image_perumated = image_tensor.permute(1, 2, 0)
    image_reshaped = image_perumated.reshape(-1, 3)
    image_pixels = np.float32(image_reshaped)
    
    scaler = MinMaxScaler()
    pixel_scaled = scaler.fit_transform(image_pixels)
    

    sample_size = min(50000, len(pixel_scaled))
    indices = np.random.choice(len(pixel_scaled), sample_size, replace=False)
    pixel_sample = pixel_scaled[indices]
    
    silhouette_scores = []
    k_range = range(2, 7)
    for k_test in k_range:
        test = KMeans(n_clusters=k_test, random_state=0, n_init=10)
        labels_test = test.fit_predict(pixel_sample)
        score = silhouette_score(pixel_sample, labels_test)
        silhouette_scores.append(score)
    
    k = k_range[np.argmax(silhouette_scores)]
    
    final_kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = final_kmeans.fit_predict(image_pixels)
    cluster_centers = final_kmeans.cluster_centers_
    
    h, w = image_perumated.shape[:2]
    segmented_mask = labels.reshape(h, w)
    
    np.random.seed(42)
    colormap = np.random.randint(0, 256, size=(k, 3), dtype=np.uint8)
    colored_mask = colormap[segmented_mask]

    original_image_pil = Image.fromarray((image_perumated.numpy() * 255).astype(np.uint8))
    colored_mask_pil = Image.fromarray(colored_mask)

    overlay_image_rgba = original_image_pil.convert("RGBA")
    mask_rgba = colored_mask_pil.convert("RGBA")

    blended = Image.blend(overlay_image_rgba, mask_rgba, alpha=0.5)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image_pil)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"Predicted Mask (k={k})")
    plt.imshow(colored_mask_pil)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(blended)
    plt.axis('off')

    plt.tight_layout()
    os.makedirs('static', exist_ok=True)
    plt.savefig("static/prediction_output.png")
    plt.close()
#predict_and_visualize("K-Means", "point.jpg", "b", "c", "d")
