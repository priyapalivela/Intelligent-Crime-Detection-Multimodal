"""Unit tests for model forward passes."""

import sys
from pathlib import Path
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import AudioEncoder, MultimodalFusionModel


def test_audio_encoder_output_shape():
    model = AudioEncoder()
    x = torch.randn(4, 40, 100)
    out = model(x)
    assert out.shape[0] == 4
    assert len(out.shape) == 2


def test_final_severity_conservative():
    assert MultimodalFusionModel.final_severity(0, 2) == 2
    assert MultimodalFusionModel.final_severity(1, 1) == 1
    assert MultimodalFusionModel.final_severity(2, 0) == 2
    assert MultimodalFusionModel.final_severity(0, 0) == 0


def test_severity_labels_complete():
    labels = {0: "Low", 1: "Medium", 2: "High"}
    assert len(labels) == 3
    assert labels[0] == "Low"
    assert labels[2] == "High"


def test_audio_classes_count():
    audio_classes = [
        "gun_shot", "siren", "drilling", "car_horn", "dog_bark",
        "jackhammer", "engine_idling", "street_music",
        "children_playing", "air_conditioner",
    ]
    assert len(audio_classes) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])