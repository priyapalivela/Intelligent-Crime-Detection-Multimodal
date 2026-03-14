# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel


class AudioEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128):
        super(AudioEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.out_dim = hidden_dim * 2

    def forward(self, x):
        if x.shape[1] != 40:
            x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        features = lstm_out.mean(dim=1)
        features = self.layer_norm(features)
        return features


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        print("Loading DistilBERT model...")
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.out_dim = 768

    def forward(self, input_ids):
        attention_mask = (input_ids != 0).long()
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden = outputs.last_hidden_state
        pooled = last_hidden.mean(dim=1)
        return pooled


class MultimodalFusionModel(nn.Module):
    def __init__(self, embed_dim=256, num_classes=3):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.text_encoder  = TextEncoder()
        self.audio_proj = nn.Linear(self.audio_encoder.out_dim, embed_dim)
        self.text_proj  = nn.Linear(self.text_encoder.out_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim * 2, num_classes)
        self.embed_dim   = embed_dim
        self.num_classes = num_classes

    def forward(self, audio, text):
        audio_feat = self.audio_encoder(audio)
        text_feat  = self.text_encoder(text)
        audio_emb  = F.relu(self.audio_proj(audio_feat))
        text_emb   = F.relu(self.text_proj(text_feat))
        fused  = torch.cat([audio_emb, text_emb], dim=1)
        logits = self.classifier(fused)
        return logits, audio_emb, text_emb

    @staticmethod
    def final_severity(audio_pred: int, text_pred: int) -> int:
        if audio_pred == 2 or text_pred == 2:
            return 2
        if audio_pred == 1 or text_pred == 1:
            return 1
        return 0
