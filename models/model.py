"""
model.py
---------
EfficientNet-B0 based transfer learning model for cotton disease detection.
Lightweight enough to train locally on CPU/GPU with a small dataset.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


# ── Disease Classes ────────────────────────────────────────────────────────────
CLASSES = [
    "Bacterial_Blight",
    "Healthy",
    "Alternaria_Leaf_Spot",
    "Curl_Virus",
    "Fusarium_Wilt",
]
NUM_CLASSES = len(CLASSES)


# ── Model Definition ───────────────────────────────────────────────────────────
class CottonDiseaseModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Freeze early layers (feature extractor)
        for name, param in self.backbone.named_parameters():
            if "features.0" in name or "features.1" in name:
                param.requires_grad = False

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── GradCAM Utility ────────────────────────────────────────────────────────────
class GradCAM:
    """Gradient-weighted Class Activation Mapping for EfficientNet-B0."""

    def __init__(self, model: CottonDiseaseModel):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.backbone.features[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> torch.Tensor:
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pool gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.squeeze()


# ── Factory Functions ──────────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES) -> CottonDiseaseModel:
    return CottonDiseaseModel(num_classes=num_classes)


def load_model(checkpoint_path: str, device: str = "cpu") -> CottonDiseaseModel:
    model = build_model()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model
