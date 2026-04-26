import pandas as pd
import shutil
from pathlib import Path
import numpy as np
from . import config

def organize_isic_dataset():
    """
    Organiza as imagens do ISIC baseado nos arquivos CSV de metadados
    """
    # Ler os arquivos CSV de metadados
    validation_df = pd.read_csv(config.VALIDATION_METADATA)
    test_df = pd.read_csv(config.TEST_METADATA)
    
    # Obter todas as categorias únicas, removendo NaN
    all_categories = set()
    for df in [validation_df, test_df]:
        categories = df['benign_malignant'].dropna().unique()
        all_categories.update(categories)
    
    print(f"Categorias encontradas: {all_categories}")

    # Criar estrutura de diretórios para todas as categorias válidas
    for split in ["train", "validation"]:
        for category in all_categories:
            if pd.notna(category):  # Verificar se não é NaN
                category_dir = config.PROCESSED_DIR / split / category
                category_dir.mkdir(parents=True, exist_ok=True)
                print(f"Criado diretório: {category_dir}")

    def copy_images(df, source_folder, destination_split):
        for _, row in df.iterrows():
            # Pular registros com categoria NaN
            if pd.isna(row['benign_malignant']):
                continue
                
            # Usar isic_id para o nome da imagem
            image_id = row['isic_id']
            # Usar benign_malignant para a classificação
            category = row['benign_malignant']
            
            # Criar o diretório da categoria se não existir
            category_dir = config.PROCESSED_DIR / destination_split / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Caminhos de origem e destino
            source_path = source_folder / f"{image_id}.jpg"
            dest_path = category_dir / f"{image_id}.jpg"
            
            # Tentar copiar a imagem
            if source_path.exists():
                shutil.copy(str(source_path), str(dest_path))
            else:
                # Tentar com extensão maiúscula
                source_path = source_folder / f"{image_id}.JPG"
                if source_path.exists():
                    shutil.copy(str(source_path), str(dest_path))
                else:
                    print(f"Imagem não encontrada: {image_id}")

    # Organizar imagens de validação
    print("\nOrganizando imagens de validação...")
    copy_images(
        validation_df, 
        config.VALIDATION_IMAGES_DIR, 
        "validation"
    )

    # Organizar imagens de teste
    print("\nOrganizando imagens de teste...")
    copy_images(
        test_df, 
        config.TEST_IMAGES_DIR, 
        "train"
    )

    # Contar e exibir estatísticas
    def count_images(path):
        counts = {}
        for category in all_categories:
            if pd.notna(category):  # Verificar se não é NaN
                category_path = path / category
                if category_path.exists():
                    counts[category] = len(list(category_path.glob("*.jpg")))
                else:
                    counts[category] = 0
        return counts

    train_counts = count_images(config.PROCESSED_DIR / "train")
    val_counts = count_images(config.PROCESSED_DIR / "validation")

    print("\nEstatísticas do Dataset:")
    print("Treino:")
    for category, count in train_counts.items():
        print(f"  {category}: {count}")
    
    print("\nValidação:")
    for category, count in val_counts.items():
        print(f"  {category}: {count}")

    # Mostrar proporção das classes
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    
    if total_train > 0:
        print("\nProporção das Classes (Treino):")
        for category, count in train_counts.items():
            print(f"  {category}: {count/total_train:.2%}")
    
    if total_val > 0:
        print("\nProporção das Classes (Validação):")
        for category, count in val_counts.items():
            print(f"  {category}: {count/total_val:.2%}")

if __name__ == "__main__":
    organize_isic_dataset()