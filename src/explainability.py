import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrar hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
        # Determinar device
        self.device = next(model.parameters()).device
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor):
        # Garantir que input_tensor está no mesmo device que o modelo
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        model_output = self.model(input_tensor)
        
        if isinstance(model_output, torch.Tensor):
            target_class = torch.argmax(model_output)
        else:
            target_class = torch.argmax(model_output[0])
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        model_output[0, target_class].backward()
        
        # Get weights
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Generate CAM
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU on CAM
        cam = torch.relu(cam)
        
        # Normalize CAM
        if torch.max(cam) > 0:
            cam = cam - torch.min(cam)
            cam = cam / torch.max(cam)
        
        return cam.cpu().numpy()

def get_target_layer(model):
    """
    Retorna a última camada convolucional do modelo
    """
    # Para ResNet18
    if hasattr(model, 'layer4'):
        return model.layer4[-1]
    
    # Para VGG
    if hasattr(model, 'features'):
        return model.features[-1]
    
    # Para outros modelos, procurar a última camada convolucional
    last_conv_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_layer = module
    
    if last_conv_layer is None:
        raise ValueError("Não foi possível encontrar uma camada convolucional no modelo")
    
    return last_conv_layer

def apply_gradcam(model, image_tensor, target_layer):
    """
    Aplica Grad-CAM em uma imagem
    """
    grad_cam = GradCAM(model, target_layer)
    try:
        cam = grad_cam.generate_cam(image_tensor)
        return cam
    except Exception as e:
        print(f"Erro ao gerar Grad-CAM: {str(e)}")
        return None

def visualize_gradcam(image_path, model, target_layer, output_dir=None):
    """
    Visualiza e salva os resultados do Grad-CAM
    """
    try:
        # Carregar e preprocessar imagem
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        # Gerar Grad-CAM
        cam = apply_gradcam(model, input_tensor, target_layer)
        if cam is None:
            print("Falha ao gerar Grad-CAM")
            return
        
        # Redimensionar CAM para tamanho da imagem original
        cam_resized = cv2.resize(cam, (299, 299))
        
        # Converter para heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Converter imagem para array numpy
        image_array = np.array(image.resize((299, 299)))
        
        # Sobrepor heatmap na imagem original
        superimposed = cv2.addWeighted(image_array, 0.7, heatmap, 0.3, 0)
        
        # Criar visualização
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_array)
        plt.title('Imagem Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cam_resized, cmap='jet')
        plt.title('Mapa de Atencao')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(superimposed)
        plt.title('Sobreposicao')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Salvar resultados se output_dir for especificado
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = Path(image_path).stem
            plt.savefig(output_dir / f'{base_name}_gradcam.png')
            
        plt.show()
        
    except Exception as e:
        print(f"Erro ao processar imagem: {str(e)}")