"""Audio helpers: ffmpeg detection, conversion to 16 kHz mono WAV, chunking, silence detection."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 20.0
# A trailing piece shorter than this is merged into the previous chunk, so the
# model never sees a 1-second scrap on its own.
MIN_TAIL_SECONDS = 5.0
# Chunks whose RMS level is below this (dBFS) are treated as silence.
SILENCE_DBFS = -45.0


class AudioError(Exception):
    """A user-facing problem with ffmpeg or the input audio."""


def ffmpeg_install_hint() -> str:
    if sys.platform == "darwin":
        return "Install it with: brew install ffmpeg"
    return "Install it with your package manager, e.g.: sudo apt install ffmpeg"


def require_ffmpeg() -> str:
    """Return the ffmpeg executable path, or raise AudioError with an install hint."""
    path = shutil.which("ffmpeg")
    if not path:
        raise AudioError(f"ffmpeg was not found on PATH. {ffmpeg_install_hint()}")
    return path


_TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$")


def parse_time(value: str) -> float:
    """Parse '90', '90.5', '1:30', '01:02:03' into seconds."""
    text = value.strip()
    match = _TIME_RE.match(text)
    if not match:
        raise ValueError(f"invalid time {value!r} (use seconds, mm:ss or hh:mm:ss)")
    parts = [p for p in match.groups() if p is not None]
    seconds = 0.0
    for i, part in enumerate(parts):
        number = float(part)
        # Every field after the first one of a colon form must be below 60.
        if i > 0 and number >= 60:
            raise ValueError(f"invalid time {value!r}: minutes and seconds must be below 60")
        seconds = seconds * 60 + number
    return seconds


def validate_range(start: float | None, end: float | None) -> None:
    if start is not None and end is not None and end <= start:
        raise ValueError("--end must be after --start")


def to_wav(
    src: Path,
    dst: Path,
    start: float | None = None,
    end: float | None = None,
) -> Path:
    """Convert any ffmpeg-readable file to 16 kHz mono 16-bit WAV, optionally trimmed."""
    ffmpeg = require_ffmpeg()
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]  # input seeking: fast, and timestamps restart at 0
    cmd += ["-i", str(src)]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0.0):.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-f", "wav", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        detail = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise AudioError(f"ffmpeg could not read {src.name}: {detail}")
    return dst


def load_wav(path: Path) -> np.ndarray:
    """Read a 16-bit mono WAV written by to_wav() as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != SAMPLE_RATE:
            raise AudioError(f"{path.name} is not 16 kHz mono 16-bit PCM")
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


@dataclass
class Chunk:
    index: int
    start: float  # seconds, relative to the analysed audio
    end: float
    rms_dbfs: float
    silent: bool

    @property
    def duration(self) -> float:
        return self.end - self.start


def chunk_bounds(
    n_samples: int,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
    min_tail_seconds: float = MIN_TAIL_SECONDS,
) -> list[tuple[int, int]]:
    """Split [0, n_samples) into ~chunk_seconds pieces; a short tail joins the previous piece."""
    if n_samples <= 0:
        return []
    size = int(round(chunk_seconds * sample_rate))
    min_tail = int(round(min_tail_seconds * sample_rate))
    bounds = [(s, min(s + size, n_samples)) for s in range(0, n_samples, size)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < min_tail:
        last = bounds.pop()
        bounds[-1] = (bounds[-1][0], last[1])
    return bounds


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return float(20 * np.log10(rms)) if rms > 0 else float("-inf")


def make_chunks(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
    silence_dbfs: float = SILENCE_DBFS,
) -> list[Chunk]:
    chunks = []
    for i, (a, b) in enumerate(chunk_bounds(len(samples), sample_rate, chunk_seconds)):
        level = rms_dbfs(samples[a:b])
        chunks.append(
            Chunk(
                index=i,
                start=a / sample_rate,
                end=b / sample_rate,
                rms_dbfs=level,
                silent=bool(level < silence_dbfs),
            )
        )
    return chunks


def format_seconds(seconds: float) -> str:
    """Format as m:ss or h:mm:ss."""
    total = int(round(seconds))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
