import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Adicionar o diretório raiz ao PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.evaluation.metrics import ModelEvaluator
from src.model import create_model
from src.preprocessing import create_data_generators
import torch

def plot_accuracy_comparison(raw_acc, balanced_acc, val_acc, save_path):
    """
    Cria um gráfico de barras comparando diferentes métricas de acurácia
    """
    accuracies = [raw_acc, balanced_acc, val_acc]
    labels = ['Acurácia Bruta', 'Acurácia Balanceada', 'Acurácia de Validação']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color=colors)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')

    plt.title('Comparação entre Diferentes Métricas de Acurácia')
    plt.ylabel('Acurácia (%)')
    plt.ylim(0, 100)  # Definir limite do eixo y de 0 a 100%
    
    # Adicionar grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Salvar gráfico
    plt.savefig(save_path)
    plt.close()

def calculate_raw_accuracy(model, loader, device):
    """
    Calcula a acurácia bruta (não balanceada)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            predicted = (outputs.squeeze() > 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return (correct / total) * 100

def test_model_evaluation():
    try:
        # Criar diretórios para logs se não existirem
        log_dir = Path('logs/evaluation')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando device: {device}")
        
        # Carregar modelo
        model = create_model()
        
        # Carregar checkpoint
        checkpoint = torch.load('models/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Imprimir informações do checkpoint
        validation_accuracy = checkpoint['val_acc']
        print(f"Modelo carregado da época: {checkpoint['epoch']}")
        print(f"Acurácia de validação salva: {validation_accuracy:.2f}%")
        
        # Criar data loader para teste
        _, test_loader = create_data_generators()
        
        # Calcular acurácia bruta
        raw_accuracy = calculate_raw_accuracy(model, test_loader, device)
        print(f"Acurácia bruta no conjunto de teste: {raw_accuracy:.2f}%")
        
        # Criar avaliador
        evaluator = ModelEvaluator(model, device)
        
        # Avaliar modelo
        print("Iniciando avaliação completa do modelo...")
        results = evaluator.evaluate_model(test_loader)
        
        # Obter acurácia balanceada
        balanced_accuracy = results['basic_metrics']['accuracy'] * 100
        print(f"Acurácia balanceada: {balanced_accuracy:.2f}%")
        
        # Gerar gráfico comparativo
        plot_accuracy_comparison(
            raw_accuracy,
            balanced_accuracy,
            validation_accuracy,
            log_dir / 'accuracy_comparison.png'
        )
        
        # Gerar visualizações
        print("Gerando visualizações...")
        evaluator.plot_all_metrics(save_dir=log_dir)
        
        # Gerar relatório detalhado
        print("Gerando relatório detalhado...")
        report_path = log_dir / 'evaluation_report.md'
        
        # Adicionar comparação de acurácias ao relatório
        additional_info = f"""
        ## Comparação de Acurácias
        
        - Acurácia de Validação: {validation_accuracy:.2f}%
        - Acurácia Bruta (Test): {raw_accuracy:.2f}%
        - Acurácia Balanceada: {balanced_accuracy:.2f}%
        
        ### Interpretação
        - A acurácia bruta é mais alta devido ao desbalanceamento das classes
        - A acurácia balanceada fornece uma visão mais realista do desempenho
        - A diferença entre as métricas indica o impacto do desbalanceamento
        """
        
        report = evaluator.generate_report(save_path=report_path, additional_info=additional_info)
        
        # Salvar métricas em formato JSON
        metrics_path = log_dir / 'metrics.json'
        evaluator.save_metrics(metrics_path)
        
        # Salvar resultados em CSV
        results_path = log_dir / 'detailed_results.csv'
        evaluator.save_detailed_results(results_path)
        
        print("\nResultados principais:")
        print(f"AUC-ROC: {results['roc_data']['auc']:.4f}")
        print(f"Acurácia Bruta: {raw_accuracy:.2f}%")
        print(f"Acurácia Balanceada: {balanced_accuracy:.2f}%")
        print(f"Acurácia de Validação: {validation_accuracy:.2f}%")
        print(f"F1-Score: {results['basic_metrics']['f1']:.4f}")
        print(f"Kappa: {results['basic_metrics']['kappa']:.4f}")
        print(f"Sensitivity: {results['per_class_metrics']['sensitivity']:.4f}")
        print(f"Specificity: {results['per_class_metrics']['specificity']:.4f}")
        
        print(f"\nAvaliação completa! Resultados salvos em: {log_dir}")
        print(f"Relatório detalhado: {report_path}")
        print(f"Métricas JSON: {metrics_path}")
        print(f"Resultados detalhados: {results_path}")
        print(f"Gráfico comparativo: {log_dir / 'accuracy_comparison.png'}")
        
    except Exception as e:
        print(f"Erro durante a avaliação: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_evaluation()