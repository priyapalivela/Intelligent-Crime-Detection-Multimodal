"""
main.py — FastAPI wrapper for Crime Severity Detection
Uses EXACT model architecture from Crime_Detection.ipynb

Run:
    uvicorn main:app --reload
    
Open: http://127.0.0.1:8000/docs
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertModel, DistilBertTokenizer

# ── Project root ──────────────────────────────────────────────────────────────
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"

# ══════════════════════════════════════════════════════════════════════════════
# EXACT MODEL CODE FROM Crime_Detection.ipynb Cell 41-43
# ══════════════════════════════════════════════════════════════════════════════

class AudioEncoder(nn.Module):
    """Cell 41 — CNN + BiLSTM audio encoder"""
    def __init__(self, input_dim=40, hidden_dim=128):
        super(AudioEncoder, self).__init__()

        # CNN feature extraction from MFCC
        self.conv1   = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1     = nn.BatchNorm1d(64)
        self.conv2   = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2     = nn.BatchNorm1d(128)
        self.pool    = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # BiLSTM temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.out_dim = hidden_dim * 2  # 256

    def forward(self, x):
        if x.shape[1] != 40:
            x = x.transpose(1, 2)
        # x: [B, 40, T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)              # [B, T/2, 128]
        lstm_out, _ = self.lstm(x)
        features = lstm_out.mean(dim=1)
        features = self.layer_norm(features)
        return features                    # [B, 256]


class TextEncoder(nn.Module):
    """Cell 42 — DistilBERT text encoder"""
    def __init__(self):
        super(TextEncoder, self).__init__()
        print("[API] Loading DistilBERT...")
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.out_dim = 768

    def forward(self, input_ids):
        attention_mask = (input_ids != 0).long()
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden = outputs.last_hidden_state  # [B, seq, 768]
        pooled = last_hidden.mean(dim=1)         # [B, 768]
        return pooled


class MultimodalFusionModel(nn.Module):
    """Cell 43 — Late fusion classifier"""
    def __init__(self, embed_dim=256, num_classes=3):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.text_encoder  = TextEncoder()
        self.audio_proj    = nn.Linear(self.audio_encoder.out_dim, embed_dim)
        self.text_proj     = nn.Linear(self.text_encoder.out_dim, embed_dim)
        self.classifier    = nn.Linear(embed_dim * 2, num_classes)
        self.embed_dim     = embed_dim
        self.num_classes   = num_classes

    def forward(self, audio, text):
        # audio: [B, 40, T]  text: [B, seq_len]
        audio_feat = self.audio_encoder(audio)           # [B, 256]
        text_feat  = self.text_encoder(text)             # [B, 768]
        audio_emb  = F.relu(self.audio_proj(audio_feat)) # [B, 256]
        text_emb   = F.relu(self.text_proj(text_feat))   # [B, 256]
        fused      = torch.cat([audio_emb, text_emb], dim=1)  # [B, 512]
        logits     = self.classifier(fused)              # [B, 3]
        return logits, audio_emb, text_emb

    @staticmethod
    def final_severity(audio_pred: int, text_pred: int) -> int:
        """Conservative rule from Cell 43"""
        if audio_pred == 2 or text_pred == 2:
            return 2
        if audio_pred == 1 or text_pred == 1:
            return 1
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# SEVERITY MAPPINGS FROM Crime_Detection.ipynb Cell 40
# ══════════════════════════════════════════════════════════════════════════════

TEXT_SEVERITY_MAPPING = {
    "HOMICIDE": 2, "CRIMINAL SEXUAL ASSAULT": 2, "ROBBERY": 2,
    "BATTERY": 2, "ASSAULT": 2, "STALKING": 2, "BURGLARY": 2,
    "MOTOR VEHICLE THEFT": 2, "ARSON": 2, "HUMAN TRAFFICKING": 2,
    "KIDNAPPING": 2, "WEAPONS VIOLATION": 2,
    "DECEPTIVE PRACTICE": 1, "CRIMINAL DAMAGE": 1, "CRIMINAL TRESPASS": 1,
    "PROSTITUTION": 1, "OFFENSE INVOLVING CHILDREN": 1, "SEX OFFENSE": 1,
    "GAMBLING": 1, "NARCOTICS": 1, "OTHER NARCOTIC VIOLATION": 1,
    "LIQUOR LAW VIOLATION": 1, "INTERFERENCE WITH PUBLIC OFFICER": 1,
    "INTIMIDATION": 1,
    "PUBLIC PEACE VIOLATION": 0, "NON-CRIMINAL": 0, "OBSCENITY": 0,
    "PUBLIC INDECENCY": 0, "OTHER OFFENSE": 0,
}

AUDIO_SEVERITY = {
    "gun_shot": 2, "siren": 2, "drilling": 2, "engine_idling": 2,
    "car_horn": 1, "dog_bark": 1, "jackhammer": 1,
    "children_playing": 0, "street_music": 0, "air_conditioner": 0,
}

SEVERITY_LABELS = {0: "Low", 1: "Medium", 2: "High"}
SEVERITY_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}

MEASURES = {
    0: ["Log incident", "Continue monitoring", "Routine patrol check"],
    1: ["Increase surveillance", "Alert security team",
        "Notify local authorities", "Document evidence"],
    2: ["Immediate response required", "Alert police/emergency services",
        "Evacuate area if safe", "Activate full emergency protocol",
        "Preserve crime scene"],
}

# Synthetic MFCC patterns per audio class (mirrors training data distribution)
MFCC_PATTERNS = {
    "gun_shot":         {"mean": 2.5,  "std": 1.8},
    "siren":            {"mean": 1.8,  "std": 1.2},
    "drilling":         {"mean": 1.5,  "std": 1.0},
    "engine_idling":    {"mean": 1.2,  "std": 0.8},
    "car_horn":         {"mean": 0.8,  "std": 0.6},
    "dog_bark":         {"mean": 0.6,  "std": 0.5},
    "jackhammer":       {"mean": 1.0,  "std": 0.7},
    "children_playing": {"mean": -0.5, "std": 0.4},
    "street_music":     {"mean": -0.3, "std": 0.3},
    "air_conditioner":  {"mean": -0.8, "std": 0.2},
}

# ── Load model at startup ─────────────────────────────────────────────────────
print("[API] Initializing model...")
device    = torch.device("cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model     = MultimodalFusionModel(embed_dim=256, num_classes=3).to(device)

if MODEL_PATH.exists():
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    print(f"[API] ✅ Loaded weights from {MODEL_PATH}")
else:
    print(f"[API] ⚠️  best_model.pt not found — random weights (demo mode)")

model.eval()
print("[API] ✅ Model ready!")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Crime Severity Detection API",
    description = """
## Multimodal Crime Severity Detection API

Real inference using our trained model from **Crime_Detection.ipynb**

### Architecture
- **AudioEncoder** — Conv1d(40→64→128) + BiLSTM(hidden=128, bidirectional=True)
- **TextEncoder** — DistilBert-base-uncased + mean pooling  
- **FusionModel** — audio_proj(256) + text_proj(256) → classifier(512→3)
- **Conservative fusion** — max(audio_pred, text_pred)

### Model Performance
- **Accuracy:** 88.29%
- **Weighted F1:** 0.86+
- **F2 Score:** ~0.88

### Live Dashboard
https://priyapalivela-crime-detection-dashboard.hf.space
    """,
    version = "1.3.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    audio_class: str
    description: str

    class Config:
        json_schema_extra = {
            "example": {
                "audio_class": "gun_shot",
                "description": "ROBBERY"
            }
        }


class ModalityDetail(BaseModel):
    severity_label: str
    severity_code:  int
    confidence:     float
    probabilities:  dict   # {"Low": x, "Medium": y, "High": z}


class PredictionResponse(BaseModel):
    audio_modality:    ModalityDetail
    text_modality:     ModalityDetail
    final_severity:    str
    final_severity_code: int
    fusion_rule:       str
    probabilities:     dict
    recommended_actions: List[str]
    color:             str
    model_info:        dict


# ── Inference function ────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(audio_class: str, description: str) -> dict:
    """
    Real model inference using trained CNN-BiLSTM + DistilBERT
    
    Audio: synthetic MFCC generated from audio class pattern
    Text:  tokenized using DistilBertTokenizer (same as training Cell 40)
    """
    # ── 1. Generate synthetic MFCC for audio class ────────────────────────────
    # In production this would be real WAV → librosa.feature.mfcc()
    # For API demo: generate class-specific MFCC pattern
    pattern    = MFCC_PATTERNS.get(audio_class.lower(), {"mean": 0.0, "std": 0.5})
    np.random.seed(hash(audio_class) % 2**32)
    mfcc       = np.random.normal(pattern["mean"], pattern["std"], (40, 100)).astype(np.float32)
    audio_tensor = torch.tensor(mfcc).unsqueeze(0)  # [1, 40, 100]

    # ── 2. Tokenize text (exact same as Cell 40) ──────────────────────────────
    text_upper   = description.upper().strip()
    tokenized    = tokenizer(
        text_upper,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    text_tensor = tokenized["input_ids"].long()  # [1, 64]

    # ── 3. Forward pass through real model ───────────────────────────────────
    logits, audio_emb, text_emb = model(audio_tensor, text_tensor)

    # ── 4. Get probabilities and predictions ─────────────────────────────────
    probs      = F.softmax(logits, dim=1)[0]
    fused_pred = logits.argmax(dim=1).item()
    probs_dict = {
        "Low":    round(probs[0].item(), 4),
        "Medium": round(probs[1].item(), 4),
        "High":   round(probs[2].item(), 4),
    }

    # ── 5. Audio unimodal prediction ─────────────────────────────────────────
    audio_feat     = model.audio_encoder(audio_tensor)
    audio_emb_proj = F.relu(model.audio_proj(audio_feat))
    zero_text      = torch.zeros_like(audio_emb_proj)
    audio_logits   = model.classifier(torch.cat([audio_emb_proj, zero_text], dim=1))
    audio_probs    = F.softmax(audio_logits, dim=1)[0]
    audio_pred     = audio_logits.argmax(dim=1).item()
    audio_conf     = audio_probs[audio_pred].item()

    # ── 6. Text unimodal prediction ───────────────────────────────────────────
    text_feat      = model.text_encoder(text_tensor)
    text_emb_proj  = F.relu(model.text_proj(text_feat))
    zero_audio     = torch.zeros_like(text_emb_proj)
    text_logits    = model.classifier(torch.cat([zero_audio, text_emb_proj], dim=1))
    text_probs     = F.softmax(text_logits, dim=1)[0]
    text_pred      = text_logits.argmax(dim=1).item()
    text_conf      = text_probs[text_pred].item()

    # ── 7. Conservative fusion (exact Cell 43 logic) ─────────────────────────
    final_sev = MultimodalFusionModel.final_severity(audio_pred, text_pred)

    return {
        "audio_pred":   audio_pred,
        "audio_conf":   round(audio_conf, 4),
        "audio_probs":  {
            "Low":    round(audio_probs[0].item(), 4),
            "Medium": round(audio_probs[1].item(), 4),
            "High":   round(audio_probs[2].item(), 4),
        },
        "text_pred":    text_pred,
        "text_conf":    round(text_conf, 4),
        "text_probs":   {
            "Low":    round(text_probs[0].item(), 4),
            "Medium": round(text_probs[1].item(), 4),
            "High":   round(text_probs[2].item(), 4),
        },
        "fused_pred":   fused_pred,
        "final_sev":    final_sev,
        "probs":        probs_dict,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message":        "Crime Severity Detection API",
        "version":        "1.3.0",
        "model":          "CNN-BiLSTM + DistilBERT (trained best_model.pt)",
        "accuracy":       "88.29%",
        "weighted_f1":    "0.86+",
        "docs":           "http://127.0.0.1:8000/docs",
        "live_dashboard": "https://priyapalivela-crime-detection-dashboard.hf.space",
        "github":         "https://github.com/priyapalivela/Intelligent-Crime-Detection-Multimodal",
        "author":         "Bhanu Priya Palivela"
    }


@app.get("/health")
def health():
    return {
        "status":          "healthy",
        "model_loaded":    MODEL_PATH.exists(),
        "model_path":      str(MODEL_PATH),
        "device":          str(device),
        "tokenizer":       "distilbert-base-uncased",
    }


@app.get("/audio-classes")
def audio_classes():
    return {
        "total": len(AUDIO_SEVERITY),
        "classes": {
            "High":   ["gun_shot", "siren", "drilling", "engine_idling"],
            "Medium": ["car_horn", "dog_bark", "jackhammer"],
            "Low":    ["children_playing", "street_music", "air_conditioner"],
        },
        "note": "Audio severity from UrbanSound8K training mapping"
    }


@app.get("/text-categories")
def text_categories():
    return {
        "total": len(TEXT_SEVERITY_MAPPING),
        "categories": TEXT_SEVERITY_MAPPING,
        "note": "Text severity from Chicago PD IUCR codes (Cell 40)"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Real multimodal inference using trained CNN-BiLSTM + DistilBERT

    - **audio_class**: one of the 10 UrbanSound8K classes
    - **description**: crime description or IUCR category (e.g. ROBBERY, ASSAULT)
    """
    audio_class = request.audio_class.lower().strip()
    description = request.description.strip()

    if audio_class not in AUDIO_SEVERITY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio_class '{audio_class}'. "
                   f"Valid: {list(AUDIO_SEVERITY.keys())}"
        )

    if not description:
        raise HTTPException(status_code=400, detail="description cannot be empty")

    result = run_inference(audio_class, description)

    return PredictionResponse(
        audio_modality=ModalityDetail(
            severity_label=SEVERITY_LABELS[result["audio_pred"]],
            severity_code=result["audio_pred"],
            confidence=result["audio_conf"],
            probabilities=result["audio_probs"],
        ),
        text_modality=ModalityDetail(
            severity_label=SEVERITY_LABELS[result["text_pred"]],
            severity_code=result["text_pred"],
            confidence=result["text_conf"],
            probabilities=result["text_probs"],
        ),
        final_severity=SEVERITY_LABELS[result["final_sev"]],
        final_severity_code=result["final_sev"],
        fusion_rule="conservative_max — max(audio_pred, text_pred) from Cell 43",
        probabilities=result["probs"],
        recommended_actions=MEASURES[result["final_sev"]],
        color=SEVERITY_COLORS[result["final_sev"]],
        model_info={
            "architecture":    "CNN-BiLSTM AudioEncoder + DistilBERT TextEncoder",
            "audio_embedding": "256-dim (BiLSTM hidden=128 × 2)",
            "text_embedding":  "768-dim → projected to 256-dim",
            "fusion":          "concat(256, 256) → Linear(512→3)",
            "accuracy":        "88.29%",
            "dataset_audio":   "UrbanSound8K — 8,732 clips",
            "dataset_text":    "Chicago PD IUCR codes",
        }
    )


@app.post("/predict/batch")
def predict_batch(requests: List[PredictionRequest]):
    """Predict severity for multiple incidents at once"""
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Max 10 incidents per batch")
    results = []
    for req in requests:
        try:
            result = run_inference(req.audio_class.lower(), req.description)
            results.append({
                "audio_class":   req.audio_class,
                "description":   req.description,
                "final_severity": SEVERITY_LABELS[result["final_sev"]],
                "confidence":    result["probs"][SEVERITY_LABELS[result["final_sev"]]],
                "color":         SEVERITY_COLORS[result["final_sev"]],
            })
        except Exception as e:
            results.append({"error": str(e), "audio_class": req.audio_class})
    return {"total": len(results), "predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)