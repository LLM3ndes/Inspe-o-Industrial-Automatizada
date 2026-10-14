import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
  sys.path.insert(0, BASE_DIR)

from data.generate_industrial_dataset import generate_metal_plate_dataset
from neurovision.dataset import MetalDefectDataset, get_transforms
from neurovision.model import build_model


def train_neurovision_pipeline() -> None:
  print("=" * 70)
  print("🔬 NEUROVISION-DEFECT: TREINAMENTO DA RESNET18 + GRAD-CAM XAI")
  print("=" * 70)

  dataset_path = os.path.join(BASE_DIR, "data", "dataset")

  # 1. Verifica/Gera Dataset
  if not os.path.exists(dataset_path):
    print("\n[!] Dataset industrial não encontrado. Gerando fotos sintéticas...")
    generate_metal_plate_dataset(base_dir=dataset_path, num_per_class=100)

  # 2. Carrega Dataset e Aplica Transformações
  print("\n[1/3] Carregando Dataset e configurando Data Augmentation...")
  train_transform, val_transform = get_transforms()
  full_dataset = MetalDefectDataset(root_dir=dataset_path, transform=train_transform)

  # Split 80% Treino / 20% Validação
  train_size = int(0.8 * len(full_dataset))
  val_size = len(full_dataset) - train_size
  train_set, val_set = random_split(full_dataset, [train_size, val_size])

  train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

  print(f" • Amostras de Treino : {train_size}")
  print(f" • Amostras de Teste : {val_size}")

  # 3. Inicializa Modelo, Função de Perda e Otimizador
  print("\n[2/3] Construindo modelo ResNet18 (Transfer Learning)...")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = build_model(num_classes=2, pretrained=True).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-4)

  # 4. Loop de Treinamento (5 épocas)
  num_epochs = 5
  print(f"\n[3/3] Iniciando Treinamento por {num_epochs} épocas no dispositivo: {device}...")

  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * images.size(0)
      _, preds = outputs.max(1)
      total += labels.size(0)
      correct += preds.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    print(f" • Época {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acurácia: {epoch_acc:.1f}%")

  # 5. Salva os Pesos Treinados
  weights_path = os.path.join(BASE_DIR, "neurovision_resnet18.pth")
  torch.save(model.state_dict(), weights_path)

  print("\n" + "=" * 70)
  print(f"✅ TREINAMENTO CONCLUÍDO! Pesos salvos em: {weights_path}")
  print("=" * 70)
  print("🚀 Para abrir a Interface Web Streamlit, execute no terminal:")
  print(" streamlit run neurovision/app.py")
  print("=" * 70 + "\n")


if __name__ == "__main__":
  train_neurovision_pipeline()