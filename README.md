---
title: Crime Detection Dashboard
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

---
title: Crime Detection Dashboard
emoji: ðŸ”
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# â­ Star this repo if you find it useful!

# \# ðŸ” Intelligent Crime Detection â€” Multimodal AI

# ðŸ“Š Multimodal Deep Learning | Crime Intelligence | Real-Time AI Monitoring

# 

# !\[Python](https://img.shields.io/badge/Python-3.10-blue)

# !\[PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)

# !\[Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

# !\[Dash](https://img.shields.io/badge/Dash-Plotly-blue)

# !\[Render](https://img.shields.io/badge/Deploy-Render-purple)

# !\[License](https://img.shields.io/badge/License-MIT-green)

# 

# > A real-time crime severity detection system that fuses \*\*audio signals\*\* and \*\*text descriptions\*\* using deep learning to classify incidents as Low, Medium, or High severity â€” with a live Dash monitoring dashboard.

# 

# ---

# 

# \## ðŸš€ Live Demo

# ðŸ”— \[crime-severity-dashboard.onrender.com](https://intelligent-crime-detection-multimodal.onrender.com) \*(Free hosting may take ~30 seconds to wake up.)\*

# 

# ---

# \## ðŸ“š Table of Contents

# \- \[Overview](#-overview)

# \- \[Model Architecture](#-model-architecture)

# \- \[Results](#-results)

# \- \[Model Performance](#-model-performance)

# \- \[Project Structure](#-project-structure)

# \- \[Datasets](#-datasets)

# \- \[Installation](#-installation)

# \- \[Training](#-training)

# \- \[Dashboard Features](#-dashboard-features)

# \- \[Deployment](#-deployment)

# \- \[Tech Stack](#-tech-stack)

# \- \[Research Contributions](#-research-contributions)

# \- \[Future Work](#-future-work)

# &nbsp; 

# ---

# 

# \## ðŸ“Œ Overview

# 

# This project builds a \*\*multimodal AI system\*\* that combines:

# \- ðŸŽµ \*\*Audio\*\* â€” UrbanSound8K environmental sounds (gunshots, sirens, drilling etc.)

# \- ðŸ“ \*\*Text\*\* â€” Chicago Police Department IUCR crime descriptions

# 

# Both modalities are fused together to predict crime severity in real time, with a fully interactive web dashboard for monitoring.

# 

# Crime monitoring systems often rely on single modalities such as textual reports or CCTV footage. 

# However, real-world incidents frequently involve multiple information sources â€” environmental sounds, 

# spoken reports, and textual descriptions.

# 

# This project investigates whether combining \*\*acoustic signals and textual crime narratives\*\* 

# through multimodal deep learning can improve the detection of \*\*incident severity\*\*.

# 

# ---

# 

# \## ðŸ§  Model Architecture

# 

# ```

# &nbsp;               â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

# &nbsp;               â”‚        Multimodal        â”‚

# &nbsp;               â”‚   Crime Severity Model   â”‚

# &nbsp;               â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

# &nbsp;                            â”‚

# &nbsp;       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

# &nbsp;       â”‚                                         â”‚

# &nbsp;â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

# &nbsp;â”‚   Audio Input â”‚                        â”‚   Text Input    â”‚

# &nbsp;â”‚ (MFCC / Mel)  â”‚                        â”‚ Crime Report    â”‚

# &nbsp;â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜                        â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜

# &nbsp;        â”‚                                         â”‚

# &nbsp; â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

# &nbsp; â”‚ CNN Layers    â”‚                       â”‚ DistilBERT       â”‚

# &nbsp; â”‚ (Feature Map) â”‚                       â”‚ Text Encoder     â”‚

# &nbsp; â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜                       â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

# &nbsp;         â”‚                                        â”‚

# &nbsp;  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

# &nbsp;  â”‚  BiLSTM      â”‚                       â”‚ Context Vector  â”‚

# &nbsp;  â”‚ Temporal     â”‚                       â”‚ (768-dim)       â”‚

# &nbsp;  â”‚ Modeling     â”‚                       â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜

# &nbsp;  â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜                                â”‚

# &nbsp;          â”‚                                 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

# &nbsp;    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                         â”‚ Linear Proj â”‚

# &nbsp;    â”‚ Audio Emb.  â”‚                         â”‚   (256)     â”‚

# &nbsp;    â”‚ 256-dim     â”‚                         â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜

# &nbsp;    â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜                                â”‚

# &nbsp;           â”‚                                 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

# &nbsp;           â”‚                                 â”‚ Text Emb.   â”‚

# &nbsp;           â”‚                                 â”‚ 256-dim     â”‚

# &nbsp;           â”‚                                 â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜

# &nbsp;           â”‚                                        â”‚

# &nbsp;           â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

# &nbsp;                          â”‚

# &nbsp;                â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

# &nbsp;                â”‚  Cross-Modal Fusion â”‚

# &nbsp;                â”‚  Attention Layer    â”‚

# &nbsp;                â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

# &nbsp;                           â”‚

# &nbsp;                  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

# &nbsp;                  â”‚  Dense Layers   â”‚

# &nbsp;                  â”‚  + Dropout      â”‚

# &nbsp;                  â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜

# &nbsp;                           â”‚

# &nbsp;                â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

# &nbsp;                â”‚  Severity Classifier â”‚

# &nbsp;                â”‚ Softmax Output       â”‚

# &nbsp;                â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

# &nbsp;                           â”‚

# &nbsp;       â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”

# &nbsp;       â”‚  Low Severity | Medium | High Severity â”‚

# &nbsp;       â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

# ```

# 

# \*\*Key components:\*\*

# \- `AudioEncoder` â€” CNN layers + BiLSTM for temporal MFCC features

# \- `TextEncoder` â€” DistilBERT pretrained transformer for crime text

# \- `MultimodalFusionModel` â€” Late fusion with projection heads

# \- `SupervisedContrastiveLoss` â€” Improves class separation

# \- \*\*Conservative Fusion Rule\*\* â€” Any modality predicting High â†’ final = High

# 

# ---

# 

# \## ðŸ“Š Results

# 

# | Metric | Score |

# |--------|-------|

# | Fused Model Accuracy | ~88% |

# | Weighted F1 Score | ~0.86 |

# | F2 Score (recall-focused) | ~0.88 |

# 

# \*\*Ablation Study:\*\*

# | Config | Accuracy | F2 Score |

# |--------|----------|----------|

# | Full Fusion (Audio + Text) | Best | Best |

# | Text Only | Medium | Medium |

# | Audio Only | Lower | Lower |

# 

# ---

# \## ðŸ“Š Model Performance

# 

# \### Training History

# !\[Training History](images/training\_history.png)

# 

# \### Confusion Matrix

# !\[Confusion Matrix](images/confusion\_matrix\_final\_method3.png)

# 

# \### ROC Curves

# !\[ROC Curves](images/roc\_curves.png)

# 

# \### Ablation Study

# !\[Ablation Results](images/ablation\_results.png)

# 

# \### Real-Time Monitoring Dashboard

# !\[Dashboard](images/real\_monitoring\_dashboard.png)

# 

# ---

# \## ðŸ”Š Audio Signal Visualisations

# 

# \### Audio Signal

# !\[Audio Signal](images/Audio\_Signal%20Picture.png)

# 

# \### MEL Frequency Cepstral Coefficients

# !\[MFCC](images/MEL%20Frequency%20Cepstral%20Coefficients%20of%20Audio.png)

# 

# \### Spectrogram of Crime Sound

# !\[Spectrogram](images/Spectrogram%20of%20Crime%20Sound.png)

# 

# ---

# \## ðŸ—‚ï¸ Project Structure

# 

# ```

# Intelligent-Crime-Detection-Multimodal/

# â”‚

# â”œâ”€â”€ app.py                  # Dash web dashboard (Render deployment)

# â”œâ”€â”€ render.yaml             # Render deployment config

# â”œâ”€â”€ requirements.txt        # Python dependencies

# â”œâ”€â”€ .gitignore

# â”œâ”€â”€ LICENSE

# â”‚

# â”œâ”€â”€ data/

# â”‚   â”œâ”€â”€ raw/

# â”‚   â”‚   â”œâ”€â”€ audio/          # UrbanSound8K .wav files (fold1â€“fold10)

# â”‚   â”‚   â”‚   â””â”€â”€ UrbanSound8K.csv

# â”‚   â”‚   â””â”€â”€ text/

# â”‚   â”‚       â””â”€â”€ Chicago\_PD\_IUCR.csv

# â”‚   â”œâ”€â”€ mfcc\_data\_with\_labels\_severity.pt

# â”‚   â””â”€â”€ text\_data.pt

# â”‚

# â”œâ”€â”€ models/

# â”‚   â””â”€â”€ best\_model.pt       # Trained model weights (Git LFS)

# â”‚

# â”œâ”€â”€ notebook/

# â”‚   â””â”€â”€ Crime\_Detection.ipynb  # Full training \& evaluation notebook

# â”‚

# â”œâ”€â”€ images/

# â”‚   â”œâ”€â”€ training\_history.png

# â”‚   â”œâ”€â”€ confusion\_matrix\_final\_method3.png

# â”‚   â”œâ”€â”€ roc\_curves.png

# â”‚   â”œâ”€â”€ ablation\_results.png

# â”‚   â”œâ”€â”€ real\_monitoring\_dashboard.png

# â”‚   â””â”€â”€ explanation\_dashboards/

# â”‚

# â”œâ”€â”€ src/

# â”‚   â”œâ”€â”€ models.py           # AudioEncoder, TextEncoder, FusionModel

# â”‚   â”œâ”€â”€ audio\_encoder.py

# â”‚   â””â”€â”€ utils/

# â”‚       â”œâ”€â”€ metrics.py

# â”‚       â””â”€â”€ visualization.py

# â”‚

# â””â”€â”€ tests/

# &nbsp;   â”œâ”€â”€ test\_models.py

# &nbsp;   â””â”€â”€ test\_data\_processing.py

# ```

# 

# ---

# 

# \## ðŸ“¦ Datasets

# 

# | Dataset | Source | Size |

# |---------|--------|------|

# | UrbanSound8K | \[urbansounddataset.weebly.com](https://urbansounddataset.weebly.com/urbansound8k.html) | 8,732 audio clips |

# | Chicago PD IUCR Codes | \[Chicago Data Portal](https://data.cityofchicago.org) | Crime descriptions |

# 

# \*\*Severity Mapping:\*\*

# 

# | Label | Audio Classes | Text Categories |

# |-------|--------------|-----------------|

# | ðŸŸ¢ Low (0) | children\_playing, street\_music, air\_conditioner | OTHER OFFENSE, PUBLIC INDECENCY |

# | ðŸŸ¡ Medium (1) | car\_horn, dog\_bark, jackhammer | THEFT, CRIMINAL DAMAGE, NARCOTICS |

# | ðŸ”´ High (2) | gun\_shot, siren, drilling, engine\_idling | HOMICIDE, ROBBERY, ASSAULT, WEAPONS |

# 

# ---

# 

# \## âš™ï¸ Installation

# 

# ```bash

# git clone https://github.com/priyapalivela/Intelligent-Crime-Detection-Multimodal.git

# cd Intelligent-Crime-Detection-Multimodal

# pip install -r requirements.txt

# ```

# 

# ---

# 

# \## ðŸƒ Run Locally

# 

# ```bash

# python app.py

# ```

# Then open \[http://localhost:8050](http://localhost:8050) in your browser.

# 

# ---

# 

# \## ðŸ‹ï¸ Training

# 

# Open and run `notebook/Crime\_Detection.ipynb` in order:

# 1\. Path setup \& data loading

# 2\. EDA â€” text and audio exploration

# 3\. MFCC preprocessing â†’ saves `mfcc\_data\_with\_labels\_severity.pt`

# 4\. Text tokenization â†’ saves `text\_data.pt`

# 5\. Model definition (AudioEncoder, TextEncoder, FusionModel)

# 6\. Training with cosine annealing + contrastive loss

# 7\. Evaluation â€” confusion matrix, ROC curves, F2 score

# 8\. Ablation study

# 9\. Explainability dashboards

# 

# ---

# 

# \## ðŸ“ˆ Dashboard Features

# 

# \- ðŸ“ \*\*Geographic Map\*\* â€” Incident predictions visualized on a simulated city map (Visakhapatnam) for demonstration purposes

# \- ðŸ“Š \*\*Severity Distribution\*\* â€” Real-time bar chart of Low/Medium/High

# \- ðŸš¨ \*\*High Severity Alerts\*\* â€” Prioritized list with recommended actions

# \- ðŸ“‹ \*\*Incidents Table\*\* â€” Full log with severity badges and details modal

# \- ðŸ” \*\*Incident Modal\*\* â€” Confidence score, precautionary measures, mini map

# \- ðŸŽšï¸ \*\*Confidence Threshold Slider\*\* â€” Filter incidents by minimum confidence level

# \- ðŸ“ˆ \*\*Time-Series Chart\*\* â€” Severity trends visualized over time

# 

# ---

# \## ðŸ“¸ Dashboard Preview

# 

# Live dashboard displaying real-time crime severity predictions and alerts.

# 

# !\[Dashboard](images/real\_monitoring\_dashboard.png)

# 

# ---

# 

# \## ðŸš€ Deployment

# 

# Deployed on \*\*Render\*\* using `gunicorn`:

# 

# ```yaml

# startCommand: gunicorn app:server --workers 1 --threads 2 --timeout 300

# ```

# 

# Environment variable:

# ```

# MODEL\_PATH=models/best\_model.pt

# ```

# 

# ---

# 

# \## ðŸ› ï¸ Tech Stack

# 

# | Component | Technology |

# |-----------|-----------|

# | Audio Features | torchaudio, MFCC, Librosa, Soundfile |

# | Text Encoding | HuggingFace DistilBERT |

# | Training | PyTorch, AdamW, CosineAnnealingLR |

# | Evaluation | scikit-learn, F1/F2, ROC-AUC |

# | Dashboard | Plotly Dash, Dash Bootstrap |

# | Deployment | Render, Gunicorn |

# 

# ---

# 

# \## ðŸ”¬ Research Contributions

# 

# This project explores \*\*multimodal fusion for crime severity prediction\*\*, combining environmental audio signals with textual crime descriptions.

# 

# Key contributions include:

# 

# â€¢ Fusion of \*\*acoustic event detection\*\* and \*\*crime narrative text\*\*  

# â€¢ Use of \*\*DistilBERT + CNN-BiLSTM hybrid architecture\*\*  

# â€¢ A \*\*conservative fusion strategy\*\* prioritizing high-risk predictions  

# â€¢ Real-time monitoring through an interactive \*\*Dash dashboard\*\*

# 

# The system demonstrates how multimodal learning can support \*\*early warning and intelligent monitoring systems\*\*.

# 

# ---

# \## ðŸ”® Future Work

# 

# â€¢ Incorporate datasets where \*\*audio and text originate from the same incident\*\* (e.g., police bodycam footage with transcripts) to improve multimodal learning quality.

# â€¢ Add \*\*attention visualization\*\* to highlight which words or audio frequencies most influenced the modelâ€™s prediction.

# â€¢ Extend the system with \*\*video-based crime detection using CCTV footage\*\* as a third modality.

# â€¢ Deploy \*\*real-time streaming inputs\*\* (audio or sensor feeds) for continuous incident monitoring.

# 

# ---

# 

# \## ðŸ“œ License

# 

# MIT License â€” see \[LICENSE](LICENSE) for details.

# 

# ---

# 

# \## ðŸ‘©â€ðŸ’» Author

# 

# \*\*Priya Palivela\*\*

# Data Science Student | AI \& Multimodal Learning Enthusiast  

# 

# \[GitHub](https://github.com/priyapalivela) Â· \[LinkedIn](https://www.linkedin.com/in/bhanu-priya-palivela-2045s/)




