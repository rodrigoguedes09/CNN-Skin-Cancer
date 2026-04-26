# src/api/app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path
import torch
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adicionar o diretório raiz ao PYTHONPATH
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from .schemas import PredictionResponse, ErrorResponse
from .utils import load_model, process_image, get_prediction

app = FastAPI(
    title="Skin Lesion Detection API",
    description="API para detecção de lesões de pele malignas",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar modelo na inicialização
model = None
device = None

@app.on_event("startup")
async def startup_event():
    global model, device
    try:
        model, device = load_model()
        logger.info("Modelo carregado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "Skin Lesion Detection API", "status": "active"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Verificar se o arquivo é uma imagem
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Arquivo enviado não é uma imagem"
            )
        
        # Ler arquivo
        contents = await file.read()
        
        # Processar imagem
        image_tensor = process_image(contents)
        
        # Fazer predição
        result = get_prediction(model, image_tensor, device)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Erro durante predição: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Erro ao processar imagem",
                detail=str(e)
            ).dict()
        )