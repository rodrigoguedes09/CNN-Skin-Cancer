import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# test_model.py
import torch
from pathlib import Path
from src.model import create_model
from src.predict import predict_image, visualize_prediction
from src import config

def test_interactive():
    # Carregar modelo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model()
    model.load_state_dict(torch.load(config.MODEL_DIR / 'best_model.pth'))
    model.to(device)
    model.eval()
    
    while True:
        # Solicitar caminho da imagem
        image_path = input("\nDigite o caminho da imagem (ou 'q' para sair): ")
        if image_path.lower() == 'q':
            break
        
        try:
            # Fazer predição
            prob = predict_image(model, image_path, device)
            print(f"\nProbabilidade de ser Maligno: {prob:.2%}")
            
            # Visualizar resultado
            visualize_prediction(image_path, prob)
            
        except Exception as e:
            print(f"Erro ao processar imagem: {e}")

if __name__ == "__main__":
    test_interactive()