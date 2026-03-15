# Changelog
All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.2.0] - 2026-03-15
### Changed
- Migrated deployment from Render to Hugging Face Spaces (Docker)
- Replaced Render-specific render.yaml with Dockerfile for containerized deployment
- Fixed OOM crash on Render free tier (512MB limit) — HF Spaces provides 16GB RAM
- Pinned all dependency versions in requirements.txt for Docker stability
- Fixed wordcloud version (1.9.0 → 1.9.2) for Python 3.11 compatibility
- Updated gunicorn port binding from 8050 → 7860 (HF Spaces requirement)

### Added
- Dockerfile with python:3.11-slim base image
- Realistic demo incident descriptions replacing placeholder text
- Git LFS tracking for all binary files (.pt, .png, .wav)

## [1.1.0] - 2026-03-11
### Added
- Crime audio visualisation images: Audio Signal, MEL Frequency Cepstral Coefficients, Spectrogram of Crime Sound
- Crime_Detection.ipynb notebook pushed to notebook/ folder (outputs cleared)

### Fixed
- Resolved merge conflict in requirements.txt, merged both versions cleanly
- Added missing dependencies: librosa, scipy, Pillow
- Removed dev-only packages from Render build: jupyter, notebook, black, flake8, opencv-python
- Fixed app.py host from 0.0.0.0 to 127.0.0.1 to resolve WinError 10049 on Windows
- Added weights_only=True to torch.load() to suppress security warning
- Increased gunicorn timeout from 120s to 300s for DistilBERT model load time on Render

---

## [1.0.0] - 2025
### Added
- Multimodal fusion model combining CNN audio encoder + DistilBERT text encoder
- MFCC feature extraction pipeline for UrbanSound8K audio dataset
- Crime description tokenisation pipeline for Chicago PD IUCR dataset
- Supervised contrastive loss with gradual DistilBERT layer unfreezing
- Conservative late-fusion severity rule (max of unimodal predictions)
- Full EDA visualizations: top crimes, pie chart, word cloud, severity distributions
- Confusion matrix with adaptive text colour, ROC curves, ablation study chart
- Per-sample explainability dashboards (modality contribution + decoded text)
- Real-time matplotlib monitoring dashboard
- Interactive Dash web dashboard (dark theme, Plotly maps, modal incident details)
- Modular src/ package: models/, data/, utils/
- Standalone scripts: train.py, evaluate.py, explainability.py, dashboard.py, monitoring_dashboard.py
- Unit tests for model forward passes and data processing utilities
- Render deployment config (app.py + gunicorn)
