# src/preprocessing.py
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
from PIL import Image
from pathlib import Path

class SkinLesionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Coletar imagens benignas
        benign_dir = self.root_dir / 'benign'
        for img_path in benign_dir.glob('*.jpg'):
            self.images.append(img_path)
            self.labels.append(0)
        
        # Coletar imagens malignas
        malignant_dir = self.root_dir / 'malignant'
        for img_path in malignant_dir.glob('*.jpg'):
            self.images.append(img_path)
            self.labels.append(1)
        
        self.labels = np.array(self.labels)

        print(f"\n[DEBUG] {self.root_dir} - Total de imagens carregadas: {len(self.images)}")
        # Opcional: imprimir os primeiros 5 arquivos para ver se os caminhos estão corretos
        print("Arquivos:", [str(path) for path in self.images[:5]])
        
        # Calcular distribuição das classes
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\nDistribuição das classes:")
        print(f"Benignas: {counts[0]}")
        print(f"Malignas: {counts[1]}")
        print(f"Proporção Benigna/Maligna: {counts[0]/counts[1]:.2f}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_data_generators():
    # Transformações para as imagens
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Criar datasets
    train_dataset = SkinLesionDataset(
        root_dir='data/processed/train',
        transform=train_transform
    )
    
    val_dataset = SkinLesionDataset(
        root_dir='data/processed/validation',
        transform=val_transform
    )
    
    # Calcular pesos para balanceamento
    class_counts = np.bincount(train_dataset.labels)
    total_samples = len(train_dataset.labels)
    
    # Calcular pesos inversos à frequência da classe
    class_weights = 1. / class_counts
    sample_weights = torch.FloatTensor([class_weights[label] for label in train_dataset.labels])
    
    # Criar sampler para balanceamento
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True
    )
    
    # Criar data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,  # Usar sampler balanceado
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader