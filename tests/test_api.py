import requests
import json
from pathlib import Path

def test_api():
    # URL da API (ajuste se necessário)
    url = "http://localhost:8000"
    
    # Testar endpoint de saúde
    health_response = requests.get(f"{url}/health")
    print("\nHealth Check:")
    print(json.dumps(health_response.json(), indent=2))
    
    # Testar predição
    image_path = "data/processed/validation/benign/exemplo.jpg"  # Ajuste o caminho
    
    with open(image_path, "rb") as f:
        files = {"file": ("image.jpg", f, "image/jpeg")}
        response = requests.post(f"{url}/predict", files=files)
    
    print("\nPrediction Response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_api()