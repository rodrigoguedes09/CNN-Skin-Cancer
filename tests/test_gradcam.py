# tests/test_gradcam.py
import sys
from pathlib import Path
import torch
from torchvision import transforms

# Adicionar o diretório raiz ao PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.model import create_model
from src.explainability import visualize_gradcam, get_target_layer

def test_gradcam():
    try:
        # Configurar device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando device: {device}")
        
        # Carregar modelo
        model = create_model()
        checkpoint = torch.load('models/best_model.pth', map_location=device)
        # Extrair apenas o state_dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Obter camada alvo
        target_layer = get_target_layer(model)
        
        # Diretório para salvar resultados
        output_dir = Path('logs/gradcam_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Testar com algumas imagens do conjunto de validação
        test_dirs = [
            Path('data/processed/validation/benign'),
            Path('data/processed/validation/malignant')
        ]
        
        for test_dir in test_dirs:
            if not test_dir.exists():
                print(f"Diretório não encontrado: {test_dir}")
                continue
            
            print(f"\nProcessando imagens de: {test_dir}")
            image_paths = list(test_dir.glob('*.jpg'))[:3]  # Testar com 3 imagens de cada classe
            
            for image_path in image_paths:
                print(f"Processando: {image_path.name}")
                try:
                    visualize_gradcam(
                        image_path,
                        model,
                        target_layer,
                        output_dir=output_dir
                    )
                    print("Sucesso!")
                except Exception as e:
                    print(f"Erro: {str(e)}")
                    
    except Exception as e:
        print(f"Erro: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gradcam()