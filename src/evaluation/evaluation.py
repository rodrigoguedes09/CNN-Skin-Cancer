import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Converter probabilidades em classes
    pred_classes = (all_preds > 0.5).astype(int)
    
    # Criar diretório para salvar resultados
    results_dir = Path("logs/evaluation_reports")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Relatório de classificação
    print("\nRelatório de Classificação:")
    class_report = classification_report(all_labels, pred_classes, 
                                      target_names=['Benigno', 'Maligno'])
    print(class_report)
    
    # Salvar relatório em arquivo
    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(class_report)
    
    # Matriz de Confusão
    cm = confusion_matrix(all_labels, pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.savefig(results_dir / 'confusion_matrix.png')
    plt.close()
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig(results_dir / 'roc_curve.png')
    plt.close()
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'roc_auc': roc_auc
    }