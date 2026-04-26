# src/api/utils.py
import torch
from PIL import Image
import io
from torchvision import transforms
import time
from pathlib import Path

def load_model():
    from src.model import create_model
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model()
    
    # Ajuste o caminho conforme necessário
    model_path = Path(__file__).parent.parent.parent / 'models' / 'best_model.pth'
    model.load_state_dict(torch.load(str(model_path)))
    model.to(device)
    model.eval()
    
    return model, device

def process_image(image_bytes):
    # Abrir imagem
    image = Image.open(io.BytesIO(image_bytes))
    
    # Converter para RGB se necessário
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Criar transformações
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Aplicar transformações
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def get_prediction(model, image_tensor, device):
    with torch.no_grad():
        start_time = time.time()
        
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probability = torch.sigmoid(outputs).cpu().numpy()[0][0]
        
        processing_time = time.time() - start_time
        
        return {
            'probability': float(probability),
            'prediction': 'Maligno' if probability > 0.5 else 'Benigno',
            'confidence': float(abs(probability - 0.5) * 2),  # Confiança baseada na distância do limiar
            'processing_time': processing_time
        }