import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

MODEL_PATH     = "models/model.pt"
MANGONET_PATH  = "models/mangonet.pth"
CLASS_NAMES    = ["Harvest Ready", "Raw (Not Ready)"]
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

_model      = None
_mangonet   = None

def load_model():
    global _model
    if _model is not None:
        return _model
    import warnings
    warnings.filterwarnings("ignore")
    _model = torch.hub.load(
        'ultralytics/yolov5', 'custom',
        path=MODEL_PATH, force_reload=False, verbose=False
    )
    _model.conf = 0.3
    _model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return _model

def load_mangonet():
    """Load MangoNet CNN for classification."""
    global _mangonet

    if _mangonet is not None:
        return _mangonet

    # MangoNet architecture (must match training definition)
    def conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    class MangoNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                conv_block(3, 32),   nn.MaxPool2d(2, 2),
                conv_block(32, 64),  nn.MaxPool2d(2, 2),
                conv_block(64, 128), nn.MaxPool2d(2, 2),
                conv_block(128, 256),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(256, 2)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = MangoNet().to(DEVICE)
    model.load_state_dict(
        torch.load(MANGONET_PATH, map_location=DEVICE)
    )
    model.eval()
    _mangonet = model
    return _mangonet


def classify_crop(crop_bgr):
    """
    MangoNet CNN classification of mango crop.
    Returns: label string, confidence float
    """
    model = load_mangonet()

    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb)
    tensor = INFER_TRANSFORM(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred  = torch.argmax(probs).item()

    return CLASS_NAMES[pred], float(probs[pred].item())

def detect_and_classify(image_path):
    """
    Full pipeline: load image -> YOLO detect -> HSV classify.

    Returns dict:
        result_img   : annotated image (BGR) for display
        detections   : list of dicts per mango found
        any_detected : bool
    """
    model = load_model()

    orig = cv2.imread(image_path)
    if orig is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = orig.shape[:2]
    result_img = orig.copy()

    # YOLO inference (expects RGB)
    rgb     = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    results = model([rgb])
    detections_raw = results.xyxyn[0]   # normalized coords

    detections = []

    if len(detections_raw) == 0:
        # No mango found — classify full image as fallback
        label, conf_cls = classify_crop(crop)
        cv2.rectangle(result_img, (2,2), (w-2,h-2), (0,165,255), 2)
        cv2.putText(result_img,
                    f"{label} (no detection)",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0,165,255), 2)
        detections.append({
            'label'     : label,
            'confidence': None,
            'box'       : None,
            'crop'      : orig.copy()
        })
        return {
            'result_img'  : result_img,
            'detections'  : detections,
            'any_detected': False
        }

    for det in detections_raw:
        x1n, y1n, x2n, y2n, conf = (
            det[0].item(), det[1].item(),
            det[2].item(), det[3].item(),
            det[4].item()
        )

        if conf < 0.3:
            continue

        x1 = max(0, int(x1n * w))
        y1 = max(0, int(y1n * h))
        x2 = min(w, int(x2n * w))
        y2 = min(h, int(y2n * h))

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue

        crop  = orig[y1:y2, x1:x2].copy()
        label, conf_cls = classify_crop(crop)

        # Color: green for harvest ready, red for raw
        # color = (0, 200, 80) if label == "Harvest Ready" else (0, 0, 255)
        color = (0, 0, 255) if label == "Harvest Ready" else (0, 0, 255)

        cv2.rectangle(result_img, (x1,y1), (x2,y2), color, 100)
        cv2.putText(
            result_img,
            f"{label}  {conf*100:.0f}%",
            (x1, max(y1-12, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85,
            color, 2
        )

        detections.append({
            'label'     : label,
            'confidence': conf,
            'box'       : (x1, y1, x2, y2),
            'crop'      : crop
        })

    return {
        'result_img'  : result_img,
        'detections'  : detections,
        'any_detected': len(detections) > 0
    }


if __name__ == "__main__":
    import sys, os
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/harvest/H1.jpg"
    os.makedirs("results", exist_ok=True)
    r = detect_and_classify(path)
    print(f"Detected: {r['any_detected']}")
    for i, d in enumerate(r['detections']):
        print(f"  Mango {i+1}: {d['label']}  conf={d['confidence']}")
    cv2.imwrite("results/test_output.jpg", r['result_img'])
    print("Saved -> results/test_output.jpg")
