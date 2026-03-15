---
title: Crime Detection Dashboard
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# ⭐ Star this repo if you find it useful!

# 🔍 Intelligent Crime Detection — Multimodal AI

📊 Multimodal Deep Learning | Crime Intelligence | Real-Time AI Monitoring

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Dash](https://img.shields.io/badge/Dash-Plotly-blue)
![Docker](https://img.shields.io/badge/Deploy-Docker-blue)
![HuggingFace](https://img.shields.io/badge/Deploy-HuggingFace-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

> A real-time crime severity detection system that fuses **audio signals** and **text descriptions** using deep learning to classify incidents as Low, Medium, or High severity — with a live Dash monitoring dashboard.

---

## 🚀 Live Demo
🔗 [priyapalivela-crime-detection-dashboard.hf.space](https://priyapalivela-crime-detection-dashboard.hf.space)

---

## 📚 Table of Contents
- [Overview](#-overview)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Datasets](#-datasets)
- [Installation](#-installation)
- [Training](#-training)
- [Dashboard Features](#-dashboard-features)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)
- [Research Contributions](#-research-contributions)
- [Future Work](#-future-work)
  
---

## 📌 Overview

This project builds a **multimodal AI system** that combines:
- 🎵 **Audio** — UrbanSound8K environmental sounds (gunshots, sirens, drilling etc.)
- 📝 **Text** — Chicago Police Department IUCR crime descriptions

Both modalities are fused together to predict crime severity in real time, with a fully interactive web dashboard for monitoring.

Crime monitoring systems often rely on single modalities such as textual reports or CCTV footage. 
However, real-world incidents frequently involve multiple information sources — environmental sounds, 
spoken reports, and textual descriptions.

This project investigates whether combining **acoustic signals and textual crime narratives** 
through multimodal deep learning can improve the detection of **incident severity**.

---

## 🧠 Model Architecture

```
                ┌──────────────────────────┐
                │        Multimodal        │
                │   Crime Severity Model   │
                └────────────┬─────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
 ┌───────────────┐                        ┌─────────────────┐
 │   Audio Input │                        │   Text Input    │
 │ (MFCC / Mel)  │                        │ Crime Report    │
 └───────┬───────┘                        └────────┬────────┘
         │                                         │
  ┌───────────────┐                       ┌──────────────────┐
  │ CNN Layers    │                       │ DistilBERT       │
  │ (Feature Map) │                       │ Text Encoder     │
  └───────┬───────┘                       └────────┬─────────┘
          │                                        │
   ┌──────────────┐                       ┌─────────────────┐
   │  BiLSTM      │                       │ Context Vector  │
   │ Temporal     │                       │ (768-dim)       │
   │ Modeling     │                       └────────┬────────┘
   └───────┬──────┘                                │
           │                                 ┌─────────────┐
     ┌─────────────┐                         │ Linear Proj │
     │ Audio Emb.  │                         │   (256)     │
     │ 256-dim     │                         └──────┬──────┘
     └──────┬──────┘                                │
            │                                 ┌─────────────┐
            │                                 │ Text Emb.   │
            │                                 │ 256-dim     │
            │                                 └──────┬──────┘
            │                                        │
            └──────────────┬─────────────────────────┘
                           │
                 ┌─────────────────────┐
                 │  Cross-Modal Fusion │
                 │  Attention Layer    │
                 └──────────┬──────────┘
                            │
                   ┌─────────────────┐
                   │  Dense Layers   │
                   │  + Dropout      │
                   └────────┬────────┘
                            │
                 ┌─────────────────────┐
                 │  Severity Classifier │
                 │ Softmax Output       │
                 └──────────┬───────────┘
                            │
        ┌────────────────────────────────────────┐
        │  Low Severity | Medium | High Severity │
        └────────────────────────────────────────┘
```

**Key components:**
- `AudioEncoder` — CNN layers + BiLSTM for temporal MFCC features
- `TextEncoder` — DistilBERT pretrained transformer for crime text
- `MultimodalFusionModel` — Late fusion with projection heads
- `SupervisedContrastiveLoss` — Improves class separation
- **Conservative Fusion Rule** — Any modality predicting High → final = High

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Fused Model Accuracy | 88.29% |
| Weighted F1 Score | 0.86+ |
| F2 Score (recall-focused) | ~0.88 |

**Ablation Study:**
| Config | Accuracy | F2 Score |
|--------|----------|----------|
| Full Fusion (Audio + Text) | Best | Best |
| Text Only | Medium | Medium |
| Audio Only | Lower | Lower |

---

## 📊 Model Performance

### Training History
![Training History](images/training_history.png)

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix_final_method3.png)

### ROC Curves
![ROC Curves](images/roc_curves.png)

### Ablation Study
![Ablation Results](images/ablation_results.png)

### Real-Time Monitoring Dashboard
![Dashboard](images/real_monitoring_dashboard.png)

---

## 🔊 Audio Signal Visualisations

### Audio Signal
![Audio Signal](images/Audio_Signal%20Picture.png)

### MEL Frequency Cepstral Coefficients
![MFCC](images/MEL%20Frequency%20Cepstral%20Coefficients%20of%20Audio.png)

### Spectrogram of Crime Sound
![Spectrogram](images/Spectrogram%20of%20Crime%20Sound.png)

---

## 🗂️ Project Structure

```
Intelligent-Crime-Detection-Multimodal/
│
├── app.py                  # Dash web dashboard
├── Dockerfile              # Docker deployment config
├── requirements.txt        # Python dependencies
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/
│   │   ├── audio/          # UrbanSound8K .wav files (fold1–fold10)
│   │   │   └── UrbanSound8K.csv
│   │   └── text/
│   │       └── Chicago_PD_IUCR.csv
│   ├── mfcc_data_with_labels_severity.pt
│   └── text_data.pt
│
├── models/
│   └── best_model.pt       # Trained model weights (Git LFS)
│
├── notebook/
│   └── Crime_Detection.ipynb  # Full training & evaluation notebook
│
├── images/
│   ├── training_history.png
│   ├── confusion_matrix_final_method3.png
│   ├── roc_curves.png
│   ├── ablation_results.png
│   ├── real_monitoring_dashboard.png
│   └── explanation_dashboards/
│
├── src/
│   ├── models.py           # AudioEncoder, TextEncoder, FusionModel
│   ├── audio_encoder.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
│
└── tests/
    ├── test_models.py
    └── test_data_processing.py
```

---

## 📦 Datasets

| Dataset | Source | Size |
|---------|--------|------|
| UrbanSound8K | [urbansounddataset.weebly.com](https://urbansounddataset.weebly.com/urbansound8k.html) | 8,732 audio clips |
| Chicago PD IUCR Codes | [Chicago Data Portal](https://data.cityofchicago.org) | Crime descriptions |

**Severity Mapping:**

| Label | Audio Classes | Text Categories |
|-------|--------------|-----------------|
| 🟢 Low (0) | children_playing, street_music, air_conditioner | OTHER OFFENSE, PUBLIC INDECENCY |
| 🟡 Medium (1) | car_horn, dog_bark, jackhammer | THEFT, CRIMINAL DAMAGE, NARCOTICS |
| 🔴 High (2) | gun_shot, siren, drilling, engine_idling | HOMICIDE, ROBBERY, ASSAULT, WEAPONS |

---

## ⚙️ Installation

```bash
git clone https://github.com/priyapalivela/Intelligent-Crime-Detection-Multimodal.git
cd Intelligent-Crime-Detection-Multimodal
pip install -r requirements.txt
```

---

## 🏃 Run Locally

```bash
python app.py
```
Then open [http://localhost:8050](http://localhost:8050) in your browser.

---

## 🏋️ Training

Open and run `notebook/Crime_Detection.ipynb` in order:
1. Path setup & data loading
2. EDA — text and audio exploration
3. MFCC preprocessing → saves `mfcc_data_with_labels_severity.pt`
4. Text tokenization → saves `text_data.pt`
5. Model definition (AudioEncoder, TextEncoder, FusionModel)
6. Training with cosine annealing + contrastive loss
7. Evaluation — confusion matrix, ROC curves, F2 score
8. Ablation study
9. Explainability dashboards

---

## 📈 Dashboard Features

- 📍 **Geographic Map** — Incident predictions visualized on a simulated city map (Visakhapatnam) for demonstration purposes
- 📊 **Severity Distribution** — Real-time bar chart of Low/Medium/High
- 🚨 **High Severity Alerts** — Prioritized list with recommended actions
- 📋 **Incidents Table** — Full log with severity badges and details modal
- 🔍 **Incident Modal** — Confidence score, precautionary measures, mini map
- 🎚️ **Confidence Threshold Slider** — Filter incidents by minimum confidence level
- 📈 **Time-Series Chart** — Severity trends visualized over time

---

## 📸 Dashboard Preview

Live dashboard displaying real-time crime severity predictions and alerts.

![Dashboard](images/real_monitoring_dashboard.png)

---

## 🚀 Deployment

Deployed on **Hugging Face Spaces** using **Docker** and `gunicorn`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "app:server", "--workers", "1", "--threads", "2", "--timeout", "300", "--bind", "0.0.0.0:7860"]
```

🔗 Live at: [priyapalivela-crime-detection-dashboard.hf.space](https://priyapalivela-crime-detection-dashboard.hf.space)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Audio Features | torchaudio, MFCC, Librosa, Soundfile |
| Text Encoding | HuggingFace DistilBERT |
| Training | PyTorch, AdamW, CosineAnnealingLR |
| Evaluation | scikit-learn, F1/F2, ROC-AUC |
| Dashboard | Plotly Dash, Dash Bootstrap |
| Containerization | Docker |
| Deployment | HuggingFace Spaces, Gunicorn |

---

## 🔬 Research Contributions

This project explores **multimodal fusion for crime severity prediction**, combining environmental audio signals with textual crime descriptions.

Key contributions include:

• Fusion of **acoustic event detection** and **crime narrative text**  
• Use of **DistilBERT + CNN-BiLSTM hybrid architecture**  
• A **conservative fusion strategy** prioritizing high-risk predictions  
• Real-time monitoring through an interactive **Dash dashboard**

The system demonstrates how multimodal learning can support **early warning and intelligent monitoring systems**.

---

## 🔮 Future Work

• Incorporate datasets where **audio and text originate from the same incident** (e.g., police bodycam footage with transcripts) to improve multimodal learning quality.
• Add **attention visualization** to highlight which words or audio frequencies most influenced the model's prediction.
• Extend the system with **video-based crime detection using CCTV footage** as a third modality.
• Deploy **real-time streaming inputs** (audio or sensor feeds) for continuous incident monitoring.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

**Bhanu Priya Palivela**
Data Science Graduate | AI & Multimodal Learning Enthusiast  

[GitHub](https://github.com/priyapalivela) · [LinkedIn](https://www.linkedin.com/in/bhanu-priya-palivela-2045s/)