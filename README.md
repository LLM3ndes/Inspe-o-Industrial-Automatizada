# 🔬 NeuroVision-Defect: Inspeção Industrial Automatizada & Grad-CAM XAI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Um framework de Visão Computacional da Indústria 4.0 projetado para o controle de qualidade automatizado em linhas de produção. A plataforma classifica defeitos em superfícies metálicas usando *Transfer Learning* (**ResNet18**) e fornece interpretabilidade espacial usando **Grad-CAM (Gradient-weighted Class Activation Mapping)** para apontar a localização exata de trincas.

---

## 📌 Principais Funcionalidades & Módulos

1. **Mecanismo de Classificação por Transfer Learning (`neurovision/model.py`)**
   - Arquitetura **ResNet18** ajustada (*fine-tuning*) pré-treinada na ImageNet.
   - Classificação binária de alta precisão: **`CLEAN (OK)`** vs. **`DEFECTIVE (Com Defeito)`**.
   - Pipeline de Aumentação de Dados (`T.RandomRotation`, `T.ColorJitter`, `T.RandomHorizontalFlip`) para lidar com iluminação ambiente de fábrica e rotações de peças.

2. **Mecanismo de IA Explicável (Grad-CAM) (`neurovision/gradcam.py`)**
   - Intercepta mapas de ativação e gradientes da última camada convolucional (`layer4.conv2`).
   - Calcula o *Global Average Pooling* sobre os gradientes do mapa de feições.
   - Gera um mapa de calor de ativação térmica (`COLORMAP_JET`) sobreposto à imagem original para explicar as decisões do modelo aos operadores de fábrica.

3. **Painel Interativo de Controle de Qualidade (`neurovision/app.py`)**
   - Desenvolvido com **Streamlit** para inspeção de imagens em tempo real (*drag-and-drop*).
   - Exibe a foto original, o mapa de calor Grad-CAM e a sobreposição térmica lado a lado.
   - Fornece pontuações de confiança e recomendações operacionais automatizadas (**APROVAR PEÇA** vs. **RETIDA / BLOQUEADA**).

---

## 📂 Arquitetura do Projeto

```text
NeuroVision-Defect/
├── main.py                     # Orquestrador do Pipeline & Treinamento do Modelo
├── requirements.txt            # Dependências do Projeto
├── README.md                   # Documentação
├── .gitignore                  # Exclusões de cache & artefatos
├── data/
│   └── generate_industrial_dataset.py # Gerador de Dataset Sintético de Superfícies Industriais
└── neurovision/
    ├── __init__.py             # Inicialização do Pacote
    ├── dataset.py              # Dataset PyTorch & Pipelines de Data Augmentation
    ├── model.py                # Arquitetura ResNet18 com Transfer Learning
    ├── gradcam.py              # Gerador de Mapa de Calor Grad-CAM XAI
    └── app.py                  # Painel Interativo Streamlit
