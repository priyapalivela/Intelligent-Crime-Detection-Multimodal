"""Unit tests for data processing utilities."""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import AUDIO_SEVERITY_MAPPING, TEXT_SEVERITY_MAP
from src.utils.metrics import compute_metrics, get_confusion_matrix


def test_audio_severity_keys():
    expected = {
        "gun_shot", "siren", "car_horn", "dog_bark", "jackhammer",
        "drilling", "children_playing", "street_music", "engine_idling", "air_conditioner",
    }
    assert set(AUDIO_SEVERITY_MAPPING.keys()) == expected


def test_audio_severity_values_range():
    for k, v in AUDIO_SEVERITY_MAPPING.items():
        assert v in (0, 1, 2), f"Invalid severity {v} for {k}"


def test_text_severity_values_range():
    for k, v in TEXT_SEVERITY_MAP.items():
        assert v in (0, 1, 2), f"Invalid severity {v} for {k}"


def test_compute_metrics_perfect():
    targets = [0, 1, 2, 0, 1, 2]
    preds   = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(targets, preds)
    assert m["accuracy"]   == 100.0
    assert m["f1_weighted"] == pytest.approx(1.0, abs=1e-6)


def test_compute_metrics_all_wrong():
    targets = [0, 0, 0]
    preds   = [1, 1, 1]
    m = compute_metrics(targets, preds)
    assert m["accuracy"] == 0.0


def test_confusion_matrix_shape():
    targets = [0, 1, 2, 0, 1, 2]
    preds   = [0, 2, 1, 0, 1, 2]
    cm = get_confusion_matrix(targets, preds)
    assert cm.shape == (3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])