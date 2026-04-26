
import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
# src/evaluation/metrics.py
import numpy as np
import torch
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score, cohen_kappa_score, balanced_accuracy_score,
    matthews_corrcoef, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap
import pandas as pd

class ModelEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.results = {}

    def evaluate_model(self, data_loader):
        """
        Avaliação completa do modelo com múltiplas métricas
        """
        # Coletar predições
        true_labels, predictions, probabilities = self._get_predictions(data_loader)
        
        # Garantir que os arrays são 1D
        true_labels = true_labels.ravel()
        predictions = predictions.ravel()
        probabilities = probabilities.ravel()

        # Calcular todas as métricas
        self.results = {
            'basic_metrics': self._calculate_basic_metrics(true_labels, predictions),
            'roc_data': self._calculate_roc(true_labels, probabilities),
            'pr_data': self._calculate_pr_curve(true_labels, probabilities),
            'confusion_matrix': confusion_matrix(true_labels, predictions),
            'class_report': classification_report(true_labels, predictions, output_dict=True),
            'calibration': self._calculate_calibration(true_labels, probabilities),
            'bootstrap_metrics': self._bootstrap_metrics(true_labels, predictions),
            'per_class_metrics': self._calculate_per_class_metrics(true_labels, predictions)
        }
        
        return self.results

    def _get_predictions(self, data_loader):
        """
        Obtém predições do modelo
        """
        self.model.eval()
        true_labels = []
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                true_labels.extend(labels.numpy())
                predictions.extend(preds)
                probabilities.extend(probs)
        
        return np.array(true_labels), np.array(predictions), np.array(probabilities)

    def _calculate_basic_metrics(self, true_labels, predictions):
        """
        Calcula métricas básicas de performance
        """
        return {
            'accuracy': balanced_accuracy_score(true_labels, predictions),
            'f1': f1_score(true_labels, predictions),
            'kappa': cohen_kappa_score(true_labels, predictions),
            'mcc': matthews_corrcoef(true_labels, predictions)
        }

    def _calculate_roc(self, true_labels, probabilities):
        """
        Calcula curva ROC e AUC
        """
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        roc_auc = auc(fpr, tpr)
        return {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

    def _calculate_pr_curve(self, true_labels, probabilities):
        """
        Calcula curva Precision-Recall
        """
        precision, recall, _ = precision_recall_curve(true_labels, probabilities)
        ap = average_precision_score(true_labels, probabilities)
        return {'precision': precision, 'recall': recall, 'ap': ap}

    def _calculate_calibration(self, true_labels, probabilities, n_bins=10):
        """
        Calcula calibração do modelo (reliability diagram)
        """
        # Garantir que os arrays são 1D
        true_labels = np.array(true_labels).ravel()
        probabilities = np.array(probabilities).ravel()
        
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probabilities, bin_edges) - 1
        
        bin_accuracies = np.zeros(n_bins)
        bin_confidences = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        for i in range(n_bins):
            bin_mask = (bin_indices == i)
            if np.any(bin_mask):
                bin_accuracies[i] = np.mean(true_labels[bin_mask])
                bin_confidences[i] = np.mean(probabilities[bin_mask])
                bin_counts[i] = np.sum(bin_mask)
        
        return {
            'accuracies': bin_accuracies,
            'confidences': bin_confidences,
            'counts': bin_counts
        }


    def _bootstrap_metrics(self, true_labels, predictions, n_iterations=1000):
        """
        Calcula intervalos de confiança usando bootstrap
        """
        # Garantir que os arrays são 1D
        true_labels = np.array(true_labels).ravel()
        predictions = np.array(predictions).ravel()
        
        # Função para calcular métricas em um único bootstrap
        def bootstrap_metrics(data):
            indices = np.random.randint(0, len(data), size=len(data))
            bootstrap_sample = data[indices]
            y_true = bootstrap_sample[:, 0]
            y_pred = bootstrap_sample[:, 1]
            
            return [
                balanced_accuracy_score(y_true, y_pred),
                f1_score(y_true, y_pred),
                cohen_kappa_score(y_true, y_pred)
            ]
        
        # Preparar dados para bootstrap
        data = np.column_stack([true_labels, predictions])
        
        # Realizar bootstrap
        results = []
        for _ in range(n_iterations):
            results.append(bootstrap_metrics(data))
        
        results = np.array(results)
        
        # Calcular intervalos de confiança
        confidence_intervals = np.percentile(results, [2.5, 97.5], axis=0)
        standard_error = np.std(results, axis=0)
        
        return {
            'confidence_intervals': confidence_intervals,
            'standard_error': standard_error,
            'metric_names': ['accuracy', 'f1', 'kappa']
        }

    def _calculate_per_class_metrics(self, true_labels, predictions):
        """
        Calcula métricas separadas para cada classe
        """
        # Garantir que os arrays são 1D
        true_labels = np.array(true_labels).ravel()
        predictions = np.array(predictions).ravel()
        
        cm = confusion_matrix(true_labels, predictions)
        
        # Para classe positiva (malignant)
        tn, fp, fn, tp = cm.ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'diagnostic_odds_ratio': (tp * tn) / (fp * fn) if (fp * fn) > 0 else 0
        }

    def plot_all_metrics(self, save_dir=None):
        """
        Gera e salva todas as visualizações
        """
        if not self.results:
            raise ValueError("Execute evaluate_model primeiro!")

        # Criar diretório para salvar plots
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Plot ROC Curve
        plt.figure(figsize=(10, 8))
        roc_data = self.results['roc_data']
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f'ROC curve (AUC = {roc_data["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend()
        if save_dir:
            plt.savefig(save_dir / 'roc_curve.png')
        plt.close()

        # Plot Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        pr_data = self.results['pr_data']
        plt.plot(pr_data['recall'], pr_data['precision'],
                label=f'PR curve (AP = {pr_data["ap"]:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        if save_dir:
            plt.savefig(save_dir / 'pr_curve.png')
        plt.close()

        # Plot Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.results['confusion_matrix'], annot=True, fmt='d',
                   cmap='Blues', xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix')
        if save_dir:
            plt.savefig(save_dir / 'confusion_matrix.png')
        plt.close()

        # Plot Calibration Curve
        plt.figure(figsize=(10, 8))
        cal_data = self.results['calibration']
        plt.plot(cal_data['confidences'], cal_data['accuracies'], 'o-',
                label='Calibration curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Plot')
        plt.legend()
        if save_dir:
            plt.savefig(save_dir / 'calibration_curve.png')
        plt.close()

    def save_metrics(self, save_path):
        """
        Salva as métricas em formato JSON
        """
        import json
        
        # Converter arrays numpy para listas
        metrics_dict = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                metrics_dict[key] = {
                    k: v.tolist() if hasattr(v, 'tolist') else v
                    for k, v in value.items()
                }
            else:
                metrics_dict[key] = value.tolist() if hasattr(value, 'tolist') else value
        
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

    def save_detailed_results(self, save_path):
        """
        Salva resultados detalhados em CSV
        """
        import pandas as pd
        
        # Criar DataFrame com todas as métricas
        metrics_dict = {
            'Metric': [],
            'Value': [],
            'Category': []
        }
        
        # Adicionar métricas básicas
        for metric, value in self.results['basic_metrics'].items():
            metrics_dict['Metric'].append(metric)
            metrics_dict['Value'].append(value)
            metrics_dict['Category'].append('Basic Metrics')
        
        # Adicionar métricas por classe
        for metric, value in self.results['per_class_metrics'].items():
            metrics_dict['Metric'].append(metric)
            metrics_dict['Value'].append(value)
            metrics_dict['Category'].append('Per Class Metrics')
        
        # Adicionar AUC-ROC
        metrics_dict['Metric'].append('AUC-ROC')
        metrics_dict['Value'].append(self.results['roc_data']['auc'])
        metrics_dict['Category'].append('ROC Analysis')
        
        # Criar e salvar DataFrame
        df = pd.DataFrame(metrics_dict)
        df.to_csv(save_path, index=False)
        
    def generate_report(self, save_path=None, additional_info=None):
        """
        Gera um relatório detalhado em formato markdown
        """
        report = []
        report.append("# Relatório de Avaliação do Modelo\n")
            
        # Data e hora da avaliação
        from datetime import datetime
        report.append(f"Data da avaliação: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Métricas básicas
        report.append("## Métricas Básicas")
        basic = self.results['basic_metrics']
        for metric, value in basic.items():
            report.append(f"- **{metric.upper()}**: {value:.4f}")
        
        # Métricas por classe
        report.append("\n## Métricas por Classe")
        per_class = self.results['per_class_metrics']
        for metric, value in per_class.items():
            report.append(f"- **{metric.upper()}**: {value:.4f}")
        
        # ROC e PR
        report.append("\n## Análise ROC e Precision-Recall")
        report.append(f"- **AUC-ROC**: {self.results['roc_data']['auc']:.4f}")
        report.append(f"- **Average Precision**: {self.results['pr_data']['ap']:.4f}")
        
        # Intervalos de confiança
        report.append("\n## Intervalos de Confiança (95%)")
        bootstrap_results = self.results['bootstrap_metrics']
        ci = bootstrap_results['confidence_intervals']
        metric_names = bootstrap_results['metric_names']
        
        for i, metric in enumerate(metric_names):
            report.append(f"- **{metric.upper()}**: ({ci[0][i]:.4f}, {ci[1][i]:.4f})")
            report.append(f"  - Erro padrão: {bootstrap_results['standard_error'][i]:.4f}")
        
        # Matriz de confusão
        report.append("\n## Matriz de Confusão")
        cm = self.results['confusion_matrix']
        report.append("```")
        report.append("          Predicted")
        report.append("Actual   Benign  Malignant")
        report.append(f"Benign     {cm[0,0]:<6d}  {cm[0,1]:<6d}")
        report.append(f"Malignant  {cm[1,0]:<6d}  {cm[1,1]:<6d}")
        report.append("```")
        
        # Salvar relatório
        report = '\n'.join(report)
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
