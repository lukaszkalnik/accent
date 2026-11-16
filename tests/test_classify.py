"""Model-free tests for the probability helpers in classify.py."""

import numpy as np
import pytest

from accent.classify import DISPLAY_NAMES, average_probabilities, display_name, softmax, top_k


def test_softmax_is_a_distribution():
    p = softmax(np.array([1.0, 2.0, 3.0]) * 30)
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 2


def test_average_is_duration_weighted():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert average_probabilities([a, b]).tolist() == [0.5, 0.5]
    assert average_probabilities([a, b], weights=[20, 5]).tolist() == pytest.approx([0.8, 0.2])


def test_average_requires_chunks():
    with pytest.raises(ValueError):
        average_probabilities([])


def test_top_k():
    labels = ["us", "england", "indian", "wales"]
    probs = np.array([0.1, 0.6, 0.25, 0.05])
    assert top_k(probs, labels) == [("england", 0.6), ("indian", 0.25), ("us", pytest.approx(0.1))]


def test_display_names_cover_all_16_labels():
    assert len(DISPLAY_NAMES) == 16
    assert display_name("southatlandtic") == "South Atlantic"
    assert display_name("unknown") == "unknown"
