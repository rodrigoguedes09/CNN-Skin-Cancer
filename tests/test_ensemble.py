from src.ensemble import EnsembleTrainer
from src.model import create_model
from src.preprocessing import create_data_generators
import torch.nn as nn
import torch
from pathlib import Path

def test_ensemble():
    # Configurar diretórios
    ensemble_dir = Path('models/ensemble')
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar data loaders
    train_loader, val_loader = create_data_generators()
    
    # Configurar ensemble
    trainer = EnsembleTrainer(
        base_model_class=create_model,
        num_models=3,  # Usar 3 modelos para teste
        model_configs=[{} for _ in range(3)]  # Configurações padrão
    )
    
    # Treinar ensemble
    criterion = nn.BCEWithLogitsLoss()
    histories = trainer.train_ensemble(
        train_loader,
        val_loader,
        criterion,
        num_epochs=5,  # Reduzido para teste
        save_dir=ensemble_dir
    )
    
    # Testar predições
    for inputs, labels in val_loader:
        predictions = trainer.predict(inputs)
        print(f"Shape das predições: {predictions.shape}")
        break
    
    print("\nEnsemble treinado e testado com sucesso!")
    
    return trainer

if __name__ == "__main__":
    test_ensemble()