# EmotiScan 🧠
### Real-Time Facial Emotion Recognition Using Deep Learning
**AI 100 — Midterm Project**

---

## What We Built
EmotiScan uses Convolutional Neural Networks (CNNs) to classify **7 human emotions** from facial images in real-time:

`Angry` | `Disgust` | `Fear` | `Happy` | `Sad` | `Surprise` | `Neutral`

We trained two models and compared them:
1. **Custom CNN** — built from scratch
2. **Transfer Learning** — fine-tuned MobileNetV2 pretrained on ImageNet

The live webcam demo lets you classify emotions in real-time during the presentation.

---

## Dataset: FER2013
- **Source:** [Kaggle — FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Size:** 35,887 grayscale 48×48 face images
- **Classes:** 7 emotion categories
- **Splits:** Training / Validation / Test

### How to Download
1. Create a free Kaggle account at kaggle.com
2. Go to the dataset page and click **Download**
3. Extract and place `fer2013.csv` inside the `data/` folder:
```
data/
└── fer2013.csv
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the notebook
Open `EmotiScan_Notebook.ipynb` in Jupyter Lab or VS Code and run all cells.

### 3. Run the live webcam demo (for presentation!)
```bash
python demo_webcam.py
```
> **Note:** You must have trained the model first (run the notebook through the "Save Model" cell).

---

## Project Structure
```
AI-100GP/
├── data/
│   └── fer2013.csv           # Download from Kaggle
├── saved_model/
│   └── emotiscan_cnn.h5      # Auto-saved after training
├── EmotiScan_Notebook.ipynb  # Main notebook: training + evaluation
├── demo_webcam.py            # Live demo for presentation
├── requirements.txt
└── README.md
```

---

## Results Summary

| Model | Test Accuracy | Parameters |
|-------|--------------|------------|
| Custom CNN | ~65% | ~2.5M |
| MobileNetV2 (TL) | ~67% | ~3.2M |

> FER2013 is a notoriously hard dataset — human accuracy is ~65% too, due to label noise.

---

## Team Members
- Devin Myers, Vina Dang

**Course:** AI 100 | **Submitted:** March 2026
