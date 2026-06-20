# 🚗 Intelligent Car Accident Claims Reviewer
### Computer Vision + Deep Learning + Google Gemini AI

---

## Project Structure

```
car_claims_ai/
├── app.py                          # Streamlit web application
├── train.py                        # Standalone training script
├── requirements.txt
├── Car_Accident_Claims_Reviewer.ipynb   # Full Jupyter walkthrough
│
├── utils/
│   ├── data_loader.py              # Dataset download & tf.data pipelines
│   ├── preprocess.py               # OpenCV preprocessing pipeline
│   └── report_generator.py         # Gemini AI report generation
│
├── models/
│   ├── model.py                    # EfficientNetB0 CNN definition
│   ├── best_model.keras            # ← created after training
│   └── class_names.json            # ← created after training
│
├── data/
│   ├── raw/                        # Kaggle download lands here
│   └── processed/
│       ├── train/{class}/
│       ├── val/{class}/
│       └── test/{class}/
│
└── outputs/
    ├── confusion_matrix.png
    ├── training_curves.png
    ├── dashboard.png
    ├── batch_results.csv
    └── sample_report.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
# Get one at: https://makersuite.google.com/app/apikey
```

### 3. Set up Kaggle credentials (for dataset download)
```bash
# Place your kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Option A: Jupyter Notebook (walkthrough)

```bash
jupyter notebook Car_Accident_Claims_Reviewer.ipynb
```

The notebook runs fully in **demo mode** (no real dataset needed) — it generates synthetic images to verify the whole pipeline. Set `USE_DEMO_DATA = False` and download from Kaggle for real training.

---

## Option B: Train from scratch

```bash
# Download dataset + train
python train.py --download --epochs 40

# Or if you already downloaded the data
python train.py --epochs 40
```

---

## Option C: Streamlit app (demo mode — no training needed)

```bash
streamlit run app.py
```

1. Open `http://localhost:8501`
2. Enable **Demo mode** in the sidebar (on by default)
3. Enter your **Gemini API key** in the sidebar
4. Upload any car damage image
5. Click **Generate Report with Gemini AI**

---

## Pipeline Details

### OpenCV Preprocessing
| Step | Method |
|------|--------|
| Resize | 224×224 px |
| Denoise | Non-local means (fastNlMeansDenoising) |
| Contrast | CLAHE on L-channel (LAB space) |
| Edge detection | Canny (50/150 thresholds) |
| Segmentation | GrabCut foreground separation |
| Region detection | Contour finding → bounding boxes |

### CNN Model — EfficientNetB0
- **Backbone**: EfficientNetB0 (ImageNet pretrained, ~5.3M params)
- **Head**: GAP → BatchNorm → Dense(256, ReLU) → Dropout(0.4) → Softmax
- **Training**: Two-phase — frozen backbone (Adam 1e-3) → fine-tune top 20 layers (Adam 1e-5)
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

### Classes
| Class | Severity | Label |
|-------|----------|-------|
| no_damage | 0 | 🟢 No Damage |
| minor_damage | 1 | 🟡 Minor Damage |
| moderate_damage | 2 | 🟠 Moderate Damage |
| severe_damage | 3 | 🔴 Severe Damage |

### LLM Report (Gemini 1.5 Flash)
The generated report contains 8 sections:
1. Executive Summary
2. Vehicle Damage Analysis
3. Severity Assessment
4. Supporting Evidence
5. Fraud Risk Assessment (LOW / MEDIUM / HIGH)
6. Repair Cost Estimate (INR)
7. Recommendation (APPROVE / APPROVE WITH INSPECTION / REFER TO ADJUSTER / REJECT)
8. Notes for Insurance Officer

---

## Dataset

**Recommended**: [Car Damage Detection — Kaggle](https://www.kaggle.com/datasets/anujms/car-damage-detection)

Expected folder structure after download:
```
data/raw/
├── 01-minor/     (or similar naming)
├── 02-moderate/
├── 03-severe/
└── 00-no_damage/
```

The `build_directory_structure()` function in `utils/data_loader.py` automatically detects class folders and splits 70/15/15.

---

## Expected Performance (CarDD full dataset)

| Metric | Expected |
|--------|----------|
| Test Accuracy | 88–93% |
| Precision | 87–91% |
| Recall | 86–90% |
| F1-Score | 87–91% |

*Actual results depend on dataset quality and training duration.*

---

## Environment

- Python 3.9+
- TensorFlow 2.12+
- CUDA GPU recommended (CPU works but slower)
- Google Gemini API key required for report generation
