# src/data/balanced_data_generators.py
from torch.utils.data import DataLoader
from torchvision import transforms
from data.balanced_dataset import MalignantAugmentedSkinLesionDataset
from src.preprocessing import SkinLesionDataset  # Sua classe original para validação

def create_data_generators(batch_size=32):
    # Transformação padrão para treinamento (para casos benignos)
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    # Transformação para validação
    val_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    train_dataset = MalignantAugmentedSkinLesionDataset(
        root_dir='data/processed/train',
        transform=train_transform,
        n_augmentations=31  
    )

    val_dataset = SkinLesionDataset(
        root_dir='data/processed/validation',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader