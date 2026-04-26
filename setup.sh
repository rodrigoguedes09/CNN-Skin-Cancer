#!/bin/bash

# Criar diretórios necessários
mkdir -p models logs data/processed/{train,validation}/{benign,malignant}

# Verificar se o modelo existe
if [ ! -f "models/best_model.pth" ]; then
    echo "AVISO: Modelo não encontrado em models/best_model.pth"
    echo "Por favor, coloque o modelo treinado neste local antes de continuar"
fi

# Construir e iniciar containers
docker-compose up --build -d

echo "Aplicação iniciada!"
echo "Interface web disponível em: http://localhost:80"
echo "API disponível em: http://localhost:80/api"