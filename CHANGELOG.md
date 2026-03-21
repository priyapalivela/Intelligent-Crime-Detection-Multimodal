# Changelog
All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.4.0] - 2026-03-21
### Added
- **Attention-based explainability** — word importance visualization in live inference panel
- Words colored by text severity prediction (red=High, orange=Medium, green=Low, grey=not a crime word)
- Non-crime filler words now shown as plain grey — no misleading highlights
- FastAPI `/explain` endpoint — returns word attention scores from 6 DistilBERT layers × 12 heads
- FastAPI `/predict/audio` endpoint — accepts real WAV files, extracts MFCC via librosa (Gap 2 closed)
- `python-multipart` dependency for WAV file upload support
- Baseline comparison table — CNN-BiLSTM + DistilBERT outperforms all single-modality baselines by 9%+

### Fixed
- Word attention section hidden when text has no crime keywords (e.g. "kite fly games")
- High Severity Alerts panel redesigned — red ID badge, gradient card, confidence display
- Location map in modal now has dark background matching dashboard theme
- Audio chart legend removed (showlegend=False) for cleaner appearance

### Changed
- Version bumped to v1.4.0
- README updated with /explain and /predict/audio endpoints
- FastAPI version string updated to 1.4.0
---

## [1.3.0] - 2026-03-15
### Added
- **Live Multimodal Inference Panel** — select audio class + type crime description → get real-time severity prediction
- Audio class dropdown with all 10 UrbanSound8K classes
- Text input box for crime description
- Conservative fusion display — shows audio severity, text severity, and final fused prediction separately
- Keyword matching engine that mirrors DistilBERT classification logic
- Matched keywords highlighted in result panel
- Recommended actions panel in inference result
- Color-coded result card (Red/Orange/Green by severity)
- Clear button to reset inference panel
<<<<<<< HEAD
- ⚡ FastAPI REST API — /predict, /health, /audio-classes, /text-categories, /predict/batch endpoints
- MLflow experiment tracking — parameters, metrics and ablation study logged
=======
>>>>>>> 8427c03342d9be33be6a48341be704d310250e1d

### Fixed
- Audio Class Distribution chart now uses discrete severity colors (Red/Orange/Green) instead of continuous color scale
- Gun Shot and Siren now correctly show Red (High) in audio chart

---

## [1.2.0] - 2026-03-15
### Changed
- Migrated deployment from Render to Hugging Face Spaces (Docker)
- Replaced render.yaml with Dockerfile for containerized deployment
- Fixed OOM crash on Render free tier (512MB limit) — HF Spaces provides 16GB RAM
- Pinned all dependency versions in requirements.txt for Docker stability
- Fixed wordcloud version (1.9.0 → 1.9.2) for Python 3.11 compatibility
- Updated gunicorn port binding from 8050 → 7860 (HF Spaces requirement)

### Added
- Dockerfile with python:3.11-slim base image
- Realistic demo incident descriptions replacing placeholder text
- Git LFS tracking for all binary files (.pt, .png, .wav)
- src/data.py with AUDIO_SEVERITY_MAPPING and TEXT_SEVERITY_MAP
- 10/10 unit tests passing (test_models.py + test_data_processing.py)
- Stats cards at top of dashboard (Total/High/Medium/Low counts)
- Audio Class Distribution chart (4th visualization)
- 20 properly matched incidents (severity correctly matched to audio class)

---

## [1.1.0] - 2026-03-11
### Added
- Crime audio visualisation images: Audio Signal, MEL Frequency Cepstral Coefficients, Spectrogram of Crime Sound
- Crime_Detection.ipynb notebook pushed to notebook/ folder (outputs cleared)

### Fixed
- Resolved merge conflict in requirements.txt
- Added missing dependencies: librosa, scipy, Pillow
- Fixed app.py host from 0.0.0.0 to 127.0.0.1 to resolve WinError 10049 on Windows
- Added weights_only=True to torch.load() to suppress security warning
- Increased gunicorn timeout from 120s to 300s for DistilBERT model load time

---

## [1.0.0] - 2025
### Added
- Multimodal fusion model combining CNN audio encoder + DistilBERT text encoder
- MFCC feature extraction pipeline for UrbanSound8K audio dataset
- Crime description tokenisation pipeline for Chicago PD IUCR dataset
- Supervised contrastive loss with gradual DistilBERT layer unfreezing
- Conservative late-fusion severity rule (max of unimodal predictions)
- Full EDA visualizations: top crimes, pie chart, word cloud, severity distributions
- Confusion matrix, ROC curves, ablation study chart
- Per-sample explainability dashboards
- Interactive Dash web dashboard (dark theme, Plotly maps, modal incident details)
- Modular src/ package: models/, data/, utils/
<<<<<<< HEAD
- Unit tests for model forward passes and data processing utilities
=======
- Unit tests for model forward passes and data processing utilities
>>>>>>> 8427c03342d9be33be6a48341be704d310250e1d
