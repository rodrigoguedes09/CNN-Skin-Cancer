# src/data/augmentation.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
import torch
from PIL import Image

class MedicalImageAugmentation:
    def __init__(self, image_size=(299, 299)):
        self.image_size = image_size
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()

    def _get_train_transforms(self):
        """
        Transformações mais complexas para treino, específicas para imagens médicas
        """
        return A.Compose([
            # Transformações geométricas
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
            
            # Transformações de cor e contraste
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.CLAHE(clip_limit=4.0, p=0.5),
                A.HueSaturationValue(p=0.5),
            ], p=0.5),
            
            # Transformações de ruído e blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MedianBlur(blur_limit=5, p=0.5),
            ], p=0.3),
            
            # Transformações específicas para imagens médicas
            A.OneOf([
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                    p=0.5
                ),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(
                    distort_limit=1.0,
                    shift_limit=0.5,
                    p=0.5
                ),
            ], p=0.3),
            
            # Normalização e conversão para tensor
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])

    def _get_val_transforms(self):
        """
        Transformações simples para validação
        """
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])

    def apply_transforms(self, image, is_training=True):
        """
        Aplica transformações em uma imagem
        """
        # Converter PIL Image para numpy array se necessário
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Garantir que a imagem está em RGB
        if len(image.shape) == 2:  # Se for grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # Se tiver canal alpha
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Aplicar transformações
        transform = self.train_transform if is_training else self.val_transform
        augmented = transform(image=image)
        return augmented['image']

def visualize_augmentations(image_path, num_examples=5):
    """
    Visualiza diferentes augmentações de uma mesma imagem
    """
    import matplotlib.pyplot as plt
    
    # Carregar imagem
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Criar augmentador
    augmentation = MedicalImageAugmentation()
    
    # Plotar imagens
    plt.figure(figsize=(20, 4))
    
    # Mostrar imagem original
    plt.subplot(1, num_examples + 1, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    
    # Mostrar augmentações
    for i in range(num_examples):
        augmented = augmentation.apply_transforms(image.copy(), is_training=True)
        # Converter tensor para numpy array
        augmented = augmented.permute(1, 2, 0).numpy()
        # Desnormalizar
        augmented = augmented * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        augmented = np.clip(augmented, 0, 1)
        
        plt.subplot(1, num_examples + 1, i + 2)
        plt.imshow(augmented)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Teste rápido
    image_path = Path("data/processed/train/benign/exemplo.jpg")  # Ajuste o caminho
    if image_path.exists():
        visualize_augmentations(image_path)
    else:
        print(f"Imagem não encontrada: {image_path}")
        # Listar algumas imagens disponíveis
        benign_dir = Path("data/processed/train/benign")
        if benign_dir.exists():
            print("\nImagens disponíveis em benign:")
            for img_path in list(benign_dir.glob("*.jpg"))[:5]:
                print(img_path)