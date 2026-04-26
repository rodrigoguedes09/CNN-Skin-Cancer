# Relatório de Avaliação do Modelo

Data da avaliação: 2025-03-26 18:58:17

## Métricas Básicas
- **ACCURACY**: 0.7309
- **F1**: 0.5787
- **KAPPA**: 0.5087
- **MCC**: 0.5146

## Métricas por Classe
- **SENSITIVITY**: 0.5124
- **SPECIFICITY**: 0.9494
- **PPV**: 0.6645
- **NPV**: 0.9088
- **DIAGNOSTIC_ODDS_RATIO**: 19.7268

## Análise ROC e Precision-Recall
- **AUC-ROC**: 0.8715
- **Average Precision**: 0.6256

## Intervalos de Confiança (95%)
- **ACCURACY**: (0.6957, 0.7653)
  - Erro padrão: 0.0174
- **F1**: (0.5193, 0.6358)
  - Erro padrão: 0.0297
- **KAPPA**: (0.4414, 0.5714)
  - Erro padrão: 0.0330

## Matriz de Confusão
```
          Predicted
Actual   Benign  Malignant
Benign     976     52    
Malignant  98      103   
```