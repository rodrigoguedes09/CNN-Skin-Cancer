# src/predict.py
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from . import config

def predict_image(model, image_path, device):
    """
    Faz predição para uma única imagem
    """
    # Carregar e preprocessar a imagem
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Fazer predição
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
    
    return prob

def visualize_prediction(image_path, probability):
    """
    Visualiza a imagem com sua predição
    """
    image = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f'Probabilidade de ser Maligno: {probability:.2%}')
    plt.axis('off')
    plt.show()