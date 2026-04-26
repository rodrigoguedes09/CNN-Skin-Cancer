import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# src/generate_report.py
import json
from datetime import datetime
from pathlib import Path
from . import config

def generate_training_report(training_history, evaluation_results):
    """
    Gera um relatório detalhado do treinamento e avaliação
    """
    report = {
        'data_treinamento': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metricas_treinamento': {
            'melhor_accuracy': float(evaluation_results['best_accuracy']),
            'auc_roc': float(evaluation_results['roc_auc'])
        },
        'parametros_modelo': {
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'image_size': config.IMAGE_SIZE
        }
    }
    
    # Salvar relatório
    report_path = config.LOGS_DIR / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)