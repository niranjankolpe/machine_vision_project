"""
predictor.py
-------------
Inference engine for cotton disease detection.
Handles single image prediction + GradCAM heatmap generation.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class Predictor:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}\n"
                "Please run `python train.py` first to train the model."
            )

        # Load classes
        classes_json = self.checkpoint_path.parent / "classes.json"
        if classes_json.exists():
            with open(classes_json) as f:
                self.classes = json.load(f)
        else:
            ckpt = torch.load(checkpoint_path, map_location=device)
            self.classes = ckpt.get("classes", [])

        # Load model
        from models.model import CottonDiseaseModel, GradCAM
        self.model = CottonDiseaseModel(num_classes=len(self.classes)).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        self.model.eval()
        self.gradcam = GradCAM(self.model)

    def predict(self, pil_image: Image.Image) -> dict:
        """
        Run inference on a PIL image.
        Returns dict with: class_name, confidence, all_probs, gradcam_overlay
        """
        # Preprocess
        tensor = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]

        pred_idx   = probs.argmax().item()
        confidence = probs[pred_idx].item()
        class_name = self.classes[pred_idx]

        all_probs = {
            self.classes[i]: round(probs[i].item() * 100, 2)
            for i in range(len(self.classes))
        }

        # GradCAM heatmap
        tensor_grad = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        tensor_grad.requires_grad_(True)
        cam = self.gradcam.generate(tensor_grad, pred_idx)
        gradcam_overlay = self._overlay_heatmap(pil_image, cam)

        return {
            "class_name":       class_name,
            "confidence":       round(confidence * 100, 2),
            "all_probs":        all_probs,
            "gradcam_overlay":  gradcam_overlay,
        }

    def _overlay_heatmap(self, pil_image: Image.Image, cam: torch.Tensor) -> Image.Image:
        """Blend GradCAM heatmap onto original image."""
        img = np.array(pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        cam_np = cam.cpu().numpy()
        cam_resized = cv2.resize(cam_np, (IMG_SIZE, IMG_SIZE))

        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img, 0.55, heatmap, 0.45, 0)
        return Image.fromarray(overlay)
