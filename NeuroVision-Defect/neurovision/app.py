import os
from PIL import Image
from neurovision.gradcam import GradCAM, overlay_cam_on_image
from neurovision.model import build_model
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T

# Configuração da página Streamlit
st.set_page_config(
    page_title="NeuroVision-Defect | Inspeção Industrial",
    page_icon="🔬",
    layout="wide",
)


# Carregamento do modelo com cache do Streamlit
@st.cache_resource
def load_trained_model(weights_path: str = "neurovision_resnet18.pth"):
  model = build_model(num_classes=2, pretrained=False)
  if os.path.exists(weights_path):
    model.load_state_dict(
        torch.load(weights_path, map_location=torch.device("cpu"))
    )
  model.eval()
  return model


st.title("🔬 NeuroVision-Defect: Inspeção de Qualidade com Visão Computacional")
st.markdown(
    "Upload de fotografia de peça metálica para **Classificação de Defeitos** e"
    " **Explicabilidade via Grad-CAM (XAI)**."
)

# Sidebar
st.sidebar.header("⚙️ Configurações do Sistema")
weights_exist = os.path.exists("neurovision_resnet18.pth")
if weights_exist:
  st.sidebar.success("✅ Modelo `neurovision_resnet18.pth` Carregado")
else:
  st.sidebar.warning(
      "⚠️ Pesos treinados não encontrados. Execute `python main.py` primeiro."
  )

alpha = st.sidebar.slider(
    "Transparência do Mapa de Calor (Grad-CAM)", 0.1, 0.9, 0.5
)

# Interface Principal de Upload
uploaded_file = st.file_uploader(
    "Selecione uma imagem (.png, .jpg)...", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")

  # Transformação para o modelo
  transform = T.Compose([
      T.Resize((224, 224)),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  input_tensor = transform(image).unsqueeze(0)

  # Inferência
  model = load_trained_model()
  target_layer = model.layer4[1].conv2
  grad_cam = GradCAM(model, target_layer)
  cam, target_class = grad_cam.generate_cam(input_tensor)
  overlay, heatmap = overlay_cam_on_image(image, cam, alpha=alpha)

  with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1).numpy()[0]

  class_names = ["CLEAN (OK)", "DEFECTIVE (Com Defeito)"]
  pred_name = class_names[target_class]
  confidence = probs[target_class] * 100

  # Exibição de Resultados em Colunas
  col1, col2, col3 = st.columns(3)

  with col1:
    st.subheader("📷 Foto Original da Peça")
    st.image(image, use_container_width=True)

  with col2:
    st.subheader("🔥 Mapa de Calor (Grad-CAM)")
    st.image(heatmap, use_container_width=True)

  with col3:
    st.subheader("🎯 Sobreposição Térmica")
    st.image(overlay, use_container_width=True)

  st.divider()

  # Cartão de Diagnóstico
  if target_class == 1:
    st.error(f"🚨 **DIAGNÓSTICO: {pred_name}** | Confiança: **{confidence:.1f}%**")
    st.warning(
        "⚠️ **Ação Recomendada:** Peça retida da esteira. A área em"
        " vermelho/amarelo indica a localização da trinca."
    )
  else:
    st.success(
        f"✅ **DIAGNÓSTICO: {pred_name}** | Confiança: **{confidence:.1f}%**"
    )
    st.info("ℹ️ **Ação Recomendada:** Peça aprovada no controle de qualidade.")