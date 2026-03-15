"""Unit tests for model forward passes."""

import sys
from pathlib import Path
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import AudioEncoder, TextEncoder, MultimodalFusionModel


def test_audio_encoder_output_shape():
    model = AudioEncoder(embedding_dim=128)
    x     = torch.randn(4, 40, 100)   # batch=4, mfcc=40, frames=100
    out   = model(x)
    assert out.shape == (4, 128), f"Expected (4,128) got {out.shape}"


def test_text_encoder_output_shape():
    model   = TextEncoder(embedding_dim=128, freeze=True)
    ids     = torch.randint(0, 1000, (4, 64))
    out     = model(ids)
    assert out.shape == (4, 128), f"Expected (4,128) got {out.shape}"


def test_fusion_model_forward():
    model  = MultimodalFusionModel()
    audio  = torch.randn(4, 40, 100)
    text   = torch.randint(0, 1000, (4, 64))
    logits, a_emb, t_emb = model(audio, text)
    assert logits.shape == (4, 3),   f"Logits shape wrong: {logits.shape}"
    assert a_emb.shape  == (4, 128), f"Audio emb shape wrong: {a_emb.shape}"
    assert t_emb.shape  == (4, 128), f"Text emb shape wrong: {t_emb.shape}"


def test_final_severity_conservative():
    assert MultimodalFusionModel.final_severity(0, 2) == 2
    assert MultimodalFusionModel.final_severity(1, 1) == 1
    assert MultimodalFusionModel.final_severity(2, 0) == 2
    assert MultimodalFusionModel.final_severity(0, 0) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
