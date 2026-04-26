# src/data/balanced_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# src/data/balanced_dataset.py

class MalignantAugmentedSkinLesionDataset(Dataset):
    """
    Dataset para lesões cutâneas que aplica data augmentation somente nas imagens malignas.
    Para cada imagem maligna, gera n_augmentations cópias com transformações aleatórias.
    Para lesões benignas, utiliza-se uma transformação padrão.
    """
    def __init__(self, root_dir, transform=None, n_augmentations=6, image_size=(299, 299)):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.n_augmentations = n_augmentations
        self.image_size = image_size  # Adicionado: Define a tamanho da imagem desejada
        
        self.image_paths = []
        self.labels = []
        
        benign_dir = self.root_dir / 'benign'
        malignant_dir = self.root_dir / 'malignant'
        
        # Para a classe benigna, adicionar cada imagem uma vez
        for img_path in benign_dir.glob('*.jpg'):
            self.image_paths.append(img_path)
            self.labels.append(0)
        
        # Para a classe maligna, adicionar cada imagem n_augmentations vezes
        for img_path in malignant_dir.glob('*.jpg'):
            for _ in range(self.n_augmentations):
                self.image_paths.append(img_path)
                self.labels.append(1)
        
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def _apply_augmentation(self, image):
        """
        Aplica transformações de data augmentation usando Albumentations.
        """
        image_np = np.array(image)
        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        aug = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(translate_percent=0.0625, scale=(0.9, 1.1), rotate=(-45, 45), p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.CLAHE(clip_limit=4.0, p=0.5),
                A.HueSaturationValue(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MedianBlur(blur_limit=5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1.0, p=0.5),
            ], p=0.3),
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        augmented = aug(image=image_np)
        return augmented['image']
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None and label == 0:
            image = self.transform(image)
        elif label == 1:
            image = self._apply_augmentation(image)
        return image, label