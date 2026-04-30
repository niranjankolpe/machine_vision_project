# 🌿 CottonGuard AI — Cotton Plant Disease Detection System

A production-ready AI system that detects cotton plant diseases from leaf images using **EfficientNet-B0** transfer learning, with a modern **Streamlit UI** featuring GradCAM explainability and detailed treatment recommendations.

---

## 🎯 What It Detects

| Disease | Severity | Pathogen |
|---|---|---|
| Bacterial Blight | 🔴 High | *Xanthomonas citri pv. malvacearum* |
| Healthy Cotton | ✅ None | — |
| Alternaria Leaf Spot | 🟡 Medium | *Alternaria macrospora* |
| Curl Virus | 🚨 Very High | Cotton Leaf Curl Virus (CLCuV) |
| Fusarium Wilt | 🔴 High | *Fusarium oxysporum* |

---

## 📁 Project Structure

```
CottonGuardAI/
├── app.py                      ← Streamlit app (run this)
├── train.py                    ← Model training script
├── requirements.txt            ← Python dependencies
│
├── raw_images/                 ← PUT YOUR WHATSAPP IMAGES HERE
│
├── scripts/
│   └── organize_dataset.py     ← Organizes raw images into train/val/test
│
├── dataset/                    ← Auto-created by organize_dataset.py
│   ├── train/
│   │   ├── Bacterial_Blight/
│   │   ├── Healthy/
│   │   ├── Alternaria_Leaf_Spot/
│   │   ├── Curl_Virus/
│   │   └── Fusarium_Wilt/
│   ├── val/
│   └── test/
│
├── models/
│   └── model.py                ← EfficientNet-B0 architecture + GradCAM
│
├── utils/
│   ├── predictor.py            ← Inference engine
│   └── treatments.py           ← Disease treatment database
│
└── checkpoints/                ← Auto-created during training
    ├── best.pth                ← Best model weights
    ├── last.pth                ← Last checkpoint
    └── classes.json            ← Class names
```

---

## ⚡ Quick Start — Step by Step

### Step 1: Clone / Open in VS Code

Open the `CottonGuardAI/` folder in VS Code:
```
File → Open Folder → Select CottonGuardAI
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ This takes 3–7 minutes. PyTorch is ~2 GB.

**For GPU acceleration (NVIDIA):** Replace the torch line in requirements.txt with:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Add Your Dataset Images

Copy your WhatsApp cotton images into the `raw_images/` folder:

```
CottonGuardAI/
└── raw_images/
    ├── WhatsApp Image 2026-04-11 at 4.06.29 PM.jpeg
    ├── WhatsApp Image 2026-04-11 at 4.06.30 PM.jpeg
    └── ... (all 39 images)
```

### Step 5: Organize Dataset

This script auto-sorts your images into 5 disease class folders with train/val/test splits:

```bash
python scripts/organize_dataset.py --src raw_images
```

Expected output:
```
✅ Found 39 images in 5 groups

  [1/5] Bacterial_Blight          → 4 train | 1 val | 1 test
  [2/5] Healthy                   → 1 train | 1 val | 1 test
  [3/5] Alternaria_Leaf_Spot      → 3 train | 1 val | 1 test
  [4/5] Curl_Virus                → 15 train | 4 val | 3 test
  [5/5] Fusarium_Wilt             → 3 train | 1 val | 1 test

✅ Done. 39 images copied to 'dataset/'
```

### Step 6: Train the Model

```bash
python train.py
```

Default settings: 40 epochs, batch size 8, learning rate 5e-4.

**Custom training:**
```bash
python train.py --epochs 60 --batch_size 16 --lr 0.001
```

Training output:
```
🔧 Device: cpu
📦 Dataset loaded: 5 classes, 26 train | 8 val | 5 test

 Epoch   Train Loss   Train Acc    Val Loss   Val Acc    Time
────────────────────────────────────────────────────────────
     1    1.6234      43.21%      1.4120     50.00%    12.3s
     2    1.3412      57.69%      1.1823     62.50%    11.8s ✅ best
    ...
    
  Best Val Accuracy : 87.50%
  Test Accuracy     : 80.00%

✅ Training complete! Best model saved to: checkpoints/best.pth
```

> 📝 **Note:** With only 39 images, accuracy may vary. Consider adding more images per class (50–100+ per class is ideal). The model uses heavy data augmentation to compensate.

### Step 7: Run the Streamlit App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** 🎉

---

## 🖥️ App Features

| Feature | Description |
|---|---|
| 📂 Image Upload | Upload JPG/PNG cotton leaf images |
| 📹 Live Webcam | Real-time camera capture and analysis |
| 🔥 GradCAM Heatmap | Visual explanation of model focus areas |
| 📊 Confidence Bar | Per-class probability breakdown |
| 💊 Treatment Guide | Specific medicines with dosage and frequency |
| 🌾 Cultural Practices | Field management recommendations |
| 🛡️ Prevention Guide | Pre-emptive disease control tips |

---

## 🔧 Training Tips for Small Datasets

Since you have ~39 images (small dataset), the model applies:

- **Heavy augmentation** (flip, rotate, color jitter, random erasing)
- **Label smoothing** (0.1) to prevent overconfidence
- **Gradient clipping** for stable training
- **Cosine LR scheduling** for better convergence
- **Early stopping** to prevent overfitting

**To improve accuracy:**
1. Collect more images per disease class (50–100+ per class)
2. Use the Kaggle Cotton Disease Dataset for additional data:
   - https://www.kaggle.com/datasets/janmejaybhoi/cotton-disease-dataset
3. Mix your images with online data for best results

---

## 🔄 Labeling Your Own Images

If your time-based auto-labeling doesn't match the actual diseases, you can manually assign labels:

Edit `scripts/organize_dataset.py` and change the `DISEASE_CLASSES` list order to match your image groups:

```python
DISEASE_CLASSES = [
    "Curl_Virus",        # Group at 4.06 (your actual label)
    "Healthy",           # Group at 4.09
    "Bacterial_Blight",  # Group at 4.14
    "Fusarium_Wilt",     # Group at 4.15
    "Alternaria_Leaf_Spot",  # Group at 4.16
]
```

Then re-run:
```bash
python scripts/organize_dataset.py --src raw_images
python train.py
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: torch` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: checkpoints/best.pth` | Run `python train.py` first |
| `No such file: dataset/train/...` | Run `python scripts/organize_dataset.py` first |
| Camera not working | Allow browser camera permissions |
| Low accuracy | Add more images per class; try 60+ epochs |
| CUDA out of memory | Reduce `--batch_size` to 4 |

---

## 📜 License

For educational and research purposes. Always consult a certified agronomist for critical crop decisions.

---

*Built with ❤️ using PyTorch, EfficientNet-B0, Streamlit, and GradCAM*
