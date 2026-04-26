# src/model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

def get_training_parameters():
    # Definir peso para a classe minoritária (ajuste este valor conforme necessário)
    pos_weight = torch.tensor([5.0])  # Aumenta a importância da classe minoritária
    
    # Criar loss function ponderada
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer_params = {
        'lr': 0.0001,
        'weight_decay': 1e-4
    }
    
    return criterion, optimizer_params