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
 
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Dash](https://img.shields.io/badge/Dash-2.17.0-blue)
![Docker](https://img.shields.io/badge/Deploy-Docker-blue)
![HuggingFace](https://img.shields.io/badge/Deploy-HuggingFace_Spaces-yellow)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF)
![Tests](https://img.shields.io/badge/Tests-10%2F10_passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
 
> A real-time crime severity detection system that fuses **audio signals** and **text descriptions** using deep learning to classify incidents as Low, Medium, or High severity — with a live interactive Dash dashboard and **real-time multimodal inference**.
 
---
 
## 🚀 Live Demo
🔗 **[priyapalivela-crime-detection-dashboard.hf.space](https://priyapalivela-crime-detection-dashboard.hf.space)**
 
---
 
## ✨ What's New in v1.3.0
- 🤖 **Live Inference Panel** — select audio class + type crime description → instant severity prediction
- 🔍 **Modality breakdown** — see audio vs text predictions separately with confidence scores
- 🏷️ **Keyword matching** — highlights which words triggered the text severity prediction
- 📊 **MLflow tracking** — experiment parameters, metrics and ablation study logged
- ⚡ **FastAPI REST API** — /predict endpoint wrapping the trained model
- 🔄 **CI/CD Pipeline** — automated testing and deployment with GitHub Actions
- 🧪 **Automated Testing** — pytest runs on every commit with code quality checks

 
---

## 📚 Table of Contents
- [Overview](#-overview)
- [Live Inference](#-live-inference)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Datasets](#-datasets)
- [Installation](#-installation)
- [Training](#-training)
- [Dashboard Features](#-dashboard-features)
- [Deployment](#-deployment)
- [REST API](#-rest-api)
- [CI/CD Pipeline](#-cicd-pipeline)
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
 
## 🤖 Live Inference
 
The dashboard includes a **live multimodal inference panel**:
 
1. Select an **audio class** (Gun Shot, Siren, Car Horn etc.)
2. Type a **crime description** in natural language
3. Click **Classify Severity**
4. See the **fused prediction** with modality breakdown
 
The system shows:
- Audio modality severity + confidence
- Text modality severity + confidence (with matched keywords)
- **Conservative fusion result** — `max(audio, text)` — minimizing false negatives
- Color-coded result card + recommended actions
 
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
├── .github/                # GitHub workflows
│   └── workflows/
│       └── ci-cd.yml       # CI/CD pipeline configuration
│
├── app.py                  # Dash web dashboard + live inference panel
├── main.py                 # FastAPI REST API wrapper
├── mlflow_test.py          # MLflow experiment tracking
├── Dockerfile              # Docker deployment config
├── requirements.txt        # Python dependencies
├── .gitignore
├── .gitattributes          # Git LFS configuration
├── LICENSE
├── CHANGELOG.md            # Version history
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
│   ├── Audio_Signal Picture.png
│   ├── MEL Frequency Cepstral Coefficients of Audio.png
│   ├── Spectrogram of Crime Sound.png
│   ├── MLflow/             # MLflow experiment tracking screenshots
│   └── explanation_dashboards/
│
├── mlruns/                 # MLflow local tracking data
│
├── src/
│   ├── models.py           # AudioEncoder, TextEncoder, FusionModel
│   ├── audio_encoder.py
│   └── utils/
│       ├── metrics.py      # compute_metrics, get_confusion_matrix
│       └── visualization.py
│
└── tests/
    ├── test_models.py           # 4 tests — all passing ✅
    └── test_data_processing.py  # 6 tests — all passing ✅
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

**Dashboard:**
```bash
python app.py
```
Then open [http://localhost:8050](http://localhost:8050)

**REST API:**
```bash
python main.py
```
Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

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
 
- 🤖 **Live Inference Panel** — type any crime description + select audio → instant prediction
- 📊 **Stats Cards** — Total/High/Medium/Low incident counts
- 📍 **Geographic Map** — Incidents on Visakhapatnam city map
- 📊 **Severity Distribution** — Bar chart Low/Medium/High
- 📈 **Severity Trends** — Time-series chart by hour
- 🔊 **Audio Distribution** — Color-coded by severity
- 🚨 **High Severity Alerts** — Prioritized list with recommended actions
- 📋 **Incidents Table** — Full log with severity badges
- 🔍 **Detail Modal** — Confidence bar, measures, mini map
- 🎚️ **Confidence Filter** — Slider to filter by minimum confidence
 
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

## ⚡ REST API

The model is also wrapped in a **FastAPI REST API** for programmatic access.
```bash
python main.py
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Model info and version |
| GET | `/health` | Model loaded status |
| GET | `/audio-classes` | All 10 audio classes with severity mapping |
| GET | `/text-categories` | All 29 IUCR crime categories |
| POST | `/predict` | Real multimodal inference |
| POST | `/predict/batch` | Batch predictions (max 10) |

### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"audio_class": "gun_shot", "description": "ROBBERY"}'
```

### Example Response
```json
{
  "audio_modality": {"severity_label": "Medium", "confidence": 0.8675},
  "text_modality": {"severity_label": "High", "confidence": 1.0},
  "final_severity": "High",
  "fusion_rule": "conservative_max — max(audio_pred, text_pred)",
  "recommended_actions": ["Immediate response required", "Alert police/emergency services"]
}
```

---

## 🔄 CI/CD Pipeline

This project uses **GitHub Actions** for continuous integration and deployment.

### Pipeline Features

- ✅ **Automated Testing** — Runs pytest on every push and pull request
- 🔍 **Code Quality Checks** — Linting with flake8 for syntax errors
- 📊 **Test Coverage** — Generates coverage reports with pytest-cov
- 🐳 **Docker Validation** — Ensures Docker image builds successfully
- 🚀 **Continuous Deployment** — Auto-deploys to Hugging Face Spaces on main branch

### Workflow Triggers

The CI/CD pipeline runs on:
- **Push** to `main` or `develop` branches
- **Pull requests** targeting `main` branch
- **Manual** workflow dispatch from Actions tab

### Pipeline Status

View real-time pipeline status and logs:
1. Go to the [Actions tab](https://github.com/priyapalivela/Intelligent-Crime-Detection-Multimodal/actions)
2. Click on any workflow run to see detailed logs
3. Green ✅ = All tests passed | Red ❌ = Failed tests

### Local Testing

Test the pipeline components locally before pushing:

```bash
# Run tests
pytest tests/ -v --cov=src

# Check code quality
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Build Docker image
docker build -t crime-detection .

# Run Docker container
docker run -p 7860:7860 crime-detection
```

---

## 🛠️ Tech Stack
 
| Component | Technology |
|-----------|-----------|
| Audio Features | torchaudio, MFCC, Librosa, Soundfile |
| Text Encoding | HuggingFace DistilBERT |
| Training | PyTorch, AdamW, CosineAnnealingLR |
| Evaluation | scikit-learn, F1/F2, ROC-AUC |
| Dashboard | Plotly Dash 2.17.0, Dash Bootstrap |
| Containerization | Docker (python:3.11-slim) |
| Deployment | HuggingFace Spaces, Gunicorn |
| Testing | pytest — 10/10 passing |
| CI/CD | GitHub Actions — automated pipeline |
| Experiment Tracking | MLflow — parameters, metrics, ablation logged |
| REST API | FastAPI + Uvicorn — /predict endpoint |
 
---
 
## 🔬 Research Contributions
 
- Fusion of **acoustic event detection** and **crime narrative text**
- **DistilBERT + CNN-BiLSTM hybrid architecture**
- **Conservative fusion strategy** prioritizing high-risk predictions
- **Live inference panel** demonstrating real multimodal classification
- Real-time monitoring through interactive **Dash dashboard**
- **Production-ready deployment** with automated CI/CD pipeline
 
---
 
## 🔮 Future Work
 
- Incorporate datasets where audio and text originate from the same incident
- Add attention visualization to highlight influential features
- Extend with video-based crime detection as a third modality
- Deploy real-time audio streaming for continuous monitoring
- Fine-tune on India-specific crime datasets (IPC codes)
- Enhance CI/CD with automated model retraining and A/B testing
 
---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests locally (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

The CI/CD pipeline will automatically run tests on your PR!

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👩‍💻 Author

**Bhanu Priya Palivela**  
Data Science Graduate | AI & Multimodal Learning Enthusiast  

[GitHub](https://github.com/priyapalivela) · [LinkedIn](https://www.linkedin.com/in/bhanu-priya-palivela-2045s/)

---

<div align="center">

**⭐ If you find this project helpful, please consider starring the repository!**

Made with ❤️ for safer communities through AI

</div>
