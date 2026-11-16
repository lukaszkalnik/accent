"""Load the CommonAccent SpeechBrain models and turn audio chunks into accent probabilities."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Models as published on the Hugging Face Hub (see their model cards).
ECAPA_SOURCE = "Jzuluaga/accent-id-commonaccent_ecapa"
XLSR_SOURCE = "Jzuluaga/accent-id-commonaccent_xlsr-en-english"

# The ECAPA classifier head returns cosine similarities, not probabilities. It was
# trained with an additive angular margin softmax of scale 30 (train_ecapa_tdnn.yaml
# in the CommonAccent repo), so softmax(30 * cosine) recovers its posterior.
ECAPA_SCALE = 30.0

DISPLAY_NAMES = {
    "african": "African",
    "australia": "Australian",
    "bermuda": "Bermudian",
    "canada": "Canadian",
    "england": "English (England)",
    "hongkong": "Hong Kong",
    "indian": "Indian",
    "ireland": "Irish",
    "malaysia": "Malaysian",
    "newzealand": "New Zealand",
    "philippines": "Filipino",
    "scotland": "Scottish",
    "singapore": "Singaporean",
    "southatlandtic": "South Atlantic",  # sic: the models' label is misspelled
    "us": "American (US)",
    "wales": "Welsh",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source: str
    description: str


MODELS = {
    "ecapa": ModelSpec("ecapa", ECAPA_SOURCE, "ECAPA-TDNN (~80 MB, fast on CPU)"),
    "xlsr": ModelSpec("xlsr", XLSR_SOURCE, "wav2vec2 XLSR-53 (~2.5 GB, slower)"),
}


class ModelError(Exception):
    """A user-facing problem loading or running a model."""


def display_name(label: str) -> str:
    return DISPLAY_NAMES.get(label, label)


def models_dir() -> Path:
    override = os.environ.get("ACCENT_MODEL_DIR")
    return Path(override) if override else Path.home() / ".cache" / "accent-cli" / "models"


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise ModelError("--device cuda was requested but CUDA is not available.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ModelError("--device mps was requested but Apple MPS is not available.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _quiet_libraries() -> None:
    # SpeechBrain gives each logger the level in SB_LOG_LEVEL when it is first imported.
    os.environ.setdefault("SB_LOG_LEVEL", "ERROR")
    warnings.filterwarnings("ignore")
    for name in ("speechbrain", "transformers", "huggingface_hub", "torch"):
        logging.getLogger(name).setLevel(logging.ERROR)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        import transformers

        transformers.logging.set_verbosity_error()
    except ImportError:
        pass


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def average_probabilities(per_chunk: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Average chunk posteriors, weighted by chunk duration so a short tail counts less."""
    if not per_chunk:
        raise ValueError("no chunks to average")
    stacked = np.stack(per_chunk)
    w = np.ones(len(per_chunk)) if weights is None else np.asarray(weights, dtype=np.float64)
    return (stacked * w[:, None]).sum(axis=0) / w.sum()


def top_k(probs: np.ndarray, labels: list[str], k: int = 3) -> list[tuple[str, float]]:
    order = np.argsort(probs)[::-1][:k]
    return [(labels[i], float(probs[i])) for i in order]


class AccentClassifier:
    def __init__(self, model: str = "ecapa", device: str = "auto"):
        if model not in MODELS:
            raise ModelError(f"unknown model {model!r}; choose from {', '.join(MODELS)}")
        self.spec = MODELS[model]
        self.device = pick_device(device)
        self._model = None
        self.labels: list[str] = []

    def load(self) -> None:
        _quiet_libraries()
        from speechbrain.utils.fetching import LocalStrategy

        # Windows needs admin rights or developer mode for symlinks; copy instead.
        strategy = LocalStrategy.COPY if sys.platform == "win32" else LocalStrategy.SYMLINK
        savedir = models_dir() / self.spec.name
        run_opts = {"device": self.device}
        try:
            if self.spec.name == "ecapa":
                from speechbrain.inference.classifiers import EncoderClassifier

                self._model = EncoderClassifier.from_hparams(
                    source=self.spec.source,
                    savedir=str(savedir),
                    run_opts=run_opts,
                    local_strategy=strategy,
                )
            else:
                from speechbrain.inference.interfaces import foreign_class

                self._model = foreign_class(
                    source=self.spec.source,
                    pymodule_file="custom_interface.py",
                    classname="CustomEncoderWav2vec2Classifier",
                    savedir=str(savedir),
                    run_opts=run_opts,
                    local_strategy=strategy,
                    # The model's yaml would otherwise create ./wav2vec2_checkpoints in the cwd.
                    overrides={"wav2vec2": {"save_path": str(models_dir() / "wav2vec2_checkpoints")}},
                )
        except OSError as exc:
            raise ModelError(
                f"could not download or load the {self.spec.name} model ({exc.__class__.__name__}: {exc}). "
                "Check your network connection (or try --ipv4)."
            ) from None
        encoder = self._model.hparams.label_encoder
        self.labels = [encoder.ind2lab[i] for i in range(len(encoder.ind2lab))]

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """Return probabilities over self.labels for one 16 kHz mono chunk."""
        import torch

        if self._model is None:
            self.load()
        wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            out_prob, _score, _index, _label = self._model.classify_batch(wav)
        scores = out_prob.detach().float().cpu().numpy().reshape(-1)
        if self.spec.name == "ecapa":
            return softmax(ECAPA_SCALE * scores)
        return scores / scores.sum()  # XLSR's head already applies softmax
