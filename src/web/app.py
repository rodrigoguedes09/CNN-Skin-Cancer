import streamlit as st
import torch
from torchvision import transforms  # Adicionar este import
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os

# Adicionar o diretório raiz ao PYTHONPATH
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from src.ensemble import EnsembleTrainer
from src.model import create_model
from src.preprocessing import create_data_generators

def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = EnsembleTrainer(
        base_model_class=create_model,
        num_models=3
    )
    trainer.load_ensemble('models/ensemble')
    
    return trainer, device

def process_image(image):
    # Redimensionar imagem
    image = image.resize((299, 299))
    
    # Criar transformações
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Aplicar transformações
    image_tensor = transform(image)
    # Adicionar dimensão do batch
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def predict(trainer, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = trainer.predict(image_tensor)
        probabilities = torch.sigmoid(outputs)
        return probabilities.cpu().numpy()[0][0]

def main():
    st.set_page_config(
        page_title="Detector de Lesões de Pele",
        page_icon="🔬",
        layout="wide"
    )

    st.title("Detector de Lesões de Pele 🔬")
    st.write("""
    Este aplicativo analisa imagens de lesões de pele e indica a probabilidade 
    de a lesão ser maligna ou benigna.
    """)

    try:
        # Carregar modelo
        model, device = load_model()

        # Interface para upload de imagem
        uploaded_file = st.file_uploader(
            "Escolha uma imagem de lesão de pele", 
            type=['png', 'jpg', 'jpeg']
        )

        if uploaded_file is not None:
            # Criar colunas para layout
            col1, col2 = st.columns(2)

            # Mostrar imagem original
            image = Image.open(uploaded_file)
            col1.subheader("Imagem Original")
            col1.image(image, use_container_width=True)  # Atualizado aqui

            try:
                # Processar imagem e fazer predição
                image_tensor = process_image(image)
                probability = predict(model, image_tensor, device)

                # Mostrar resultados
                col2.subheader("Resultados da Análise")
                
                # Criar um medidor visual
                col2.progress(float(probability))
                
                # Mostrar probabilidade em porcentagem
                prob_percentage = probability * 100
                result_color = "red" if prob_percentage > 50 else "green"
                
                col2.markdown(f"""
                <h3 style='color: {result_color}'>
                Probabilidade de ser maligno: {prob_percentage:.1f}%
                </h3>
                """, unsafe_allow_html=True)

                # Interpretação do resultado
                if prob_percentage > 50:
                    col2.warning("""
                    ⚠️ ATENÇÃO: Há indicação de possível malignidade.
                    Recomenda-se consultar um profissional de saúde.
                    """)
                else:
                    col2.success("""
                    ✅ A lesão apresenta características predominantemente benignas.
                    Mantenha o acompanhamento regular com seu médico.
                    """)

            except Exception as e:
                col2.error(f"Erro ao processar a imagem: {str(e)}")
                st.exception(e)  # Isso mostrará o traceback completo

    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        st.exception(e)  # Isso mostrará o traceback completo

    # Adicionar disclaimer
    st.markdown("""
    ---
    **Disclaimer:** Esta é uma ferramenta de auxílio e não substitui 
    o diagnóstico profissional. Sempre consulte um médico especialista.
    """)

    # Adicionar informações sobre o modelo
    with st.expander("ℹ️ Sobre o Modelo"):
        st.write("""
        Este modelo foi treinado com imagens do dataset ISIC e utiliza uma 
        arquitetura ResNet18 com transfer learning. O modelo foi treinado para 
        identificar lesões de pele malignas e benignas.
        
        **Notas importantes:**
        - O modelo tem uma acurácia de aproximadamente 87% no conjunto de validação
        - As imagens devem ser claras e bem focadas
        - Diferentes condições de iluminação podem afetar os resultados
        """)

if __name__ == "__main__":
    main()