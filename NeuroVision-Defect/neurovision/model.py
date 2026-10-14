import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
  """Carrega a arquitetura ResNet18 com pesos pré-treinados na ImageNet e adapta a camada final de classificação para 2 classes (OK vs DEFECTIVE)."""
  weights = ResNet18_Weights.DEFAULT if pretrained else None
  model = models.resnet18(weights=weights)

  # Extrai o número de entradas da camada linear final original (1000 classes)
  num_ftrs = model.fc.in_features

  # Substitui a camada final por uma nova camada Linear para num_classes
  model.fc = nn.Linear(num_ftrs, num_classes)

  return model