

# src/config.py
from pathlib import Path

# Diretórios base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Diretórios de dados
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

# Arquivos de metadados
VALIDATION_METADATA = DATA_DIR / "validation_metadata.csv"
TEST_METADATA = DATA_DIR / "test_metadata.csv"

# Diretórios de imagens
VALIDATION_IMAGES_DIR = DATA_DIR / "ISIC-images-validation"
TEST_IMAGES_DIR = DATA_DIR / "ISIC-images-test"

# Parâmetros do modelo
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_CLASSES = 2

# Parâmetros de aumento de dados
DATA_AUGMENTATION = {
    'rotation_range': 180,
    'horizontal_flip': True,
    'vertical_flip': True,
    'fill_mode': 'nearest',
    'zoom_range': 0.2,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2
}

