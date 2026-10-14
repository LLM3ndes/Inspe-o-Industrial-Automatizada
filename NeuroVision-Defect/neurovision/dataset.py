import os
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class MetalDefectDataset(Dataset):
  """Dataset PyTorch customizado para carregar fotos de peças metálicas e mapear os rótulos: clean = 0 (OK), defective = 1 (DEFECTIVE)."""

  def __init__(self, root_dir: str, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.samples = []
    # Mapeamento numérico de classes
    self.class_to_idx = {"clean": 0, "defective": 1}

    for class_name, label in self.class_to_idx.items():
      class_dir = os.path.join(root_dir, class_name)
      if os.path.exists(class_dir):
        for fname in os.listdir(class_dir):
          if fname.endswith((".png", ".jpg", ".jpeg")):
            self.samples.append((os.path.join(class_dir, fname), label))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
    img_path, label = self.samples[idx]
    image = Image.open(img_path).convert("RGB")
    if self.transform:
      image = self.transform(image)
    return image, label


def get_transforms() -> tuple[T.Compose, T.Compose]:
  """Retorna os pipelines de transformação e normalização ImageNet para os conjuntos de Treino e Validação."""
  train_transform = T.Compose([
      T.Resize((224, 224)),
      T.RandomHorizontalFlip(p=0.5),
      T.RandomRotation(degrees=15),
      T.ColorJitter(brightness=0.2, contrast=0.2),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  val_transform = T.Compose([
      T.Resize((224, 224)),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  return train_transform, val_transform