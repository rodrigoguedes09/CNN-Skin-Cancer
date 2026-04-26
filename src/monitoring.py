# src/monitoring.py
import mlflow
import wandb

def setup_tracking():
    """
    Configura tracking de experimentos
    """
    mlflow.start_run()
    wandb.init(project="skin-lesion-detection")

def log_metrics(metrics):
    """
    Registra métricas do modelo
    """
    mlflow.log_metrics(metrics)
    wandb.log(metrics)