import cv2
import numpy as np
import torch


class GradCAM:
  """Classe para gerar mapas de calor explicativos (Grad-CAM) a partir das ativações da última camada convolucional da ResNet18."""

  def __init__(self, model, target_layer):
    self.model = model
    self.target_layer = target_layer
    self.gradients = None
    self.activations = None

    # Registra hooks no PyTorch para capturar ativações e gradientes
    self.target_layer.register_forward_hook(self._save_activations)
    self.target_layer.register_full_backward_hook(self._save_gradients)

  def _save_activations(self, module, input, output):
    self.activations = output

  def _save_gradients(self, module, grad_input, grad_output):
    self.gradients = grad_output[0]

  def generate_cam(
      self, input_tensor: torch.Tensor, target_class: int | None = None
  ) -> tuple[np.ndarray, int]:
    self.model.eval()

    # Forward pass para extrair a pontuação das classes
    output = self.model(input_tensor)
    if target_class is None:
      target_class = output.argmax(dim=1).item()

    # Zera gradientes residuais e executa backward pass focado na classe alvo
    self.model.zero_grad()
    score = output[0, target_class]
    score.backward()

    # Extrai gradientes e ativações capturados pelos hooks
    gradients = self.gradients.data.cpu().numpy()[0]
    activations = self.activations.data.cpu().numpy()[0]

    # Cálculo dos pesos via Global Average Pooling dos gradientes
    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
      cam += w * activations[i, :, :]

    # Aplica ReLU para manter apenas feições positivas e normaliza entre 0 e 1
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
      cam = cam / cam.max()

    cam = cv2.resize(cam, (224, 224))
    return cam, target_class


def overlay_cam_on_image(
    original_image, cam: np.ndarray, alpha: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
  """Sobrepõe o mapa de calor Grad-CAM na imagem original da peça em formato RGB."""
  heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
  original_np = np.array(original_image.resize((224, 224)))
  overlay = cv2.addWeighted(original_np, 1 - alpha, heatmap, alpha, 0)
  return overlay, heatmap