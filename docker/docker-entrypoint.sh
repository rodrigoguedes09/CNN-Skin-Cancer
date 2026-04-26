#!/bin/bash

# Verificar qual serviço iniciar
if [ "$SERVICE" = "streamlit" ]; then
    echo "Iniciando Streamlit..."
    streamlit run src/web/app.py
elif [ "$SERVICE" = "api" ]; then
    echo "Iniciando FastAPI..."
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
else
    echo "Serviço não especificado. Use SERVICE=streamlit ou SERVICE=api"
    exit 1
fi