import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
# test_augmentation.py
from src.augmentation import visualize_augmentations
from pathlib import Path

def test_augmentations():
    # Procurar por uma imagem válida
    train_dir = Path("data/processed/train/benign")
    
    if not train_dir.exists():
        print(f"Diretório não encontrado: {train_dir}")
        return
    
    # Pegar a primeira imagem disponível
    image_paths = list(train_dir.glob("*.jpg"))
    if not image_paths:
        print("Nenhuma imagem encontrada")
        return
    
    image_path = image_paths[0]
    print(f"Usando imagem: {image_path}")
    
    # Visualizar diferentes augmentações
    try:
        visualize_augmentations(image_path, num_examples=5)
    except Exception as e:
        print(f"Erro ao visualizar augmentações: {str(e)}")

if __name__ == "__main__":
    test_augmentations()