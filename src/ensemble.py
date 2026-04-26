import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Inicializa o modelo ensemble
        
        Args:
            models: Lista de modelos para o ensemble
            weights: Pesos para cada modelo (opcional)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights else [1/len(models)] * len(models)
        assert len(self.weights) == len(models), "Número de pesos deve ser igual ao número de modelos"
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do ensemble
        """
        outputs = []
        for model, weight in zip(self.models, self.weights):
            output = model(x) * weight
            outputs.append(output)
        
        # Média ponderada das predições
        return torch.stack(outputs).mean(dim=0)

class EnsembleTrainer:
    def __init__(self, 
                 base_model_class,
                 num_models: int = 5,
                 model_configs: Optional[List[Dict]] = None):
        """
        Inicializa o treinador de ensemble
        
        Args:
            base_model_class: Classe do modelo base
            num_models: Número de modelos no ensemble
            model_configs: Lista de configurações para cada modelo
        """
        self.base_model_class = base_model_class
        self.num_models = num_models
        self.model_configs = model_configs or [{}] * num_models
        self.models = []
        self.ensemble = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train_model(self, 
                   model: nn.Module, 
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader,
                   criterion: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   num_epochs: int) -> Dict:
        """
        Treina um único modelo
        """
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            # Modo treino
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Modo validação
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels.float())
                    val_loss += loss.item()
                    
                    predicted = (outputs.squeeze() > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            # Calcular métricas
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            
            # Guardar histórico
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logger.info(f'Epoch {epoch+1}/{num_epochs}:')
            logger.info(f'Train Loss: {train_loss:.4f}')
            logger.info(f'Val Loss: {val_loss:.4f}')
            logger.info(f'Val Acc: {val_acc:.2f}%')
            
            # Salvar melhor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
        return history
    
    def train_ensemble(self,
                      train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader,
                      criterion: nn.Module,
                      num_epochs: int,
                      save_dir: Optional[str] = None) -> Dict:
        """
        Treina o ensemble completo
        """
        histories = []
        save_dir = Path(save_dir) if save_dir else Path('models/ensemble')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(self.num_models):
            logger.info(f'\nTreinando modelo {i+1}/{self.num_models}')
            
            # Criar e mover modelo para device
            model = self.base_model_class(**self.model_configs[i]).to(self.device)
            optimizer = torch.optim.Adam(model.parameters())
            
            # Treinar modelo
            history = self.train_model(
                model, train_loader, val_loader, criterion, optimizer, num_epochs
            )
            histories.append(history)
            
            # Salvar modelo
            torch.save(model.state_dict(), save_dir / f'model_{i+1}.pth')
            self.models.append(model)
        
        # Criar ensemble
        self.ensemble = EnsembleModel(self.models)
        
        # Salvar histórico
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(histories, f)
        
        return histories
    
    def load_ensemble(self, models_dir: str):
        """
        Carrega um ensemble salvo
        """
        models_dir = Path(models_dir)
        self.models = []
        
        for i in range(self.num_models):
            model = self.base_model_class(**self.model_configs[i]).to(self.device)
            model.load_state_dict(torch.load(models_dir / f'model_{i+1}.pth'))
            self.models.append(model)
        
        self.ensemble = EnsembleModel(self.models)
    
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Faz predição usando o ensemble
        """
        if self.ensemble is None:
            raise ValueError("Ensemble não foi treinado ou carregado ainda")
        
        self.ensemble.eval()
        with torch.no_grad():
            return self.ensemble(input_tensor)