import shutil
import wave

import numpy as np
import pytest

from accent import audio
from accent.audio import SAMPLE_RATE, chunk_bounds, format_seconds, make_chunks, parse_time


# --- time-range parsing -------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("0", 0.0),
        ("90", 90.0),
        ("12.5", 12.5),
        ("1:30", 90.0),
        ("01:30", 90.0),
        ("0:05", 5.0),
        ("1:02:03", 3723.0),
        ("10:00.5", 600.5),
        (" 2:00 ", 120.0),
    ],
)
def test_parse_time_valid(text, expected):
    assert parse_time(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "abc", "1:2:3:4", "-5", "1:60", "1:75:00", "1::2", "1,5"])
def test_parse_time_invalid(text):
    with pytest.raises(ValueError):
        parse_time(text)


def test_validate_range():
    audio.validate_range(None, None)
    audio.validate_range(10, 20)
    audio.validate_range(None, 5)
    with pytest.raises(ValueError):
        audio.validate_range(20, 10)
    with pytest.raises(ValueError):
        audio.validate_range(5, 5)


def test_format_seconds():
    assert format_seconds(5) == "0:05"
    assert format_seconds(600) == "10:00"
    assert format_seconds(3723) == "1:02:03"


# --- chunking -----------------------------------------------------------------

def secs(bounds):
    return [(a / SAMPLE_RATE, b / SAMPLE_RATE) for a, b in bounds]


def test_chunk_empty():
    assert chunk_bounds(0) == []


def test_chunk_shorter_than_one_chunk():
    assert secs(chunk_bounds(7 * SAMPLE_RATE)) == [(0, 7)]


def test_chunk_exact_multiple():
    assert secs(chunk_bounds(60 * SAMPLE_RATE)) == [(0, 20), (20, 40), (40, 60)]


def test_chunk_keeps_long_tail():
    assert secs(chunk_bounds(48 * SAMPLE_RATE)) == [(0, 20), (20, 40), (40, 48)]


def test_chunk_merges_short_tail():
    # A 2 s tail is too short to classify alone, so it joins the last chunk.
    assert secs(chunk_bounds(42 * SAMPLE_RATE)) == [(0, 20), (20, 42)]


def test_chunk_covers_every_sample():
    n = 123_457
    bounds = chunk_bounds(n, chunk_seconds=1.0)
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    assert all(prev[1] == nxt[0] for prev, nxt in zip(bounds, bounds[1:]))


# --- silence skipping ---------------------------------------------------------

def tone(seconds, amplitude=0.3, freq=220.0):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_silent_chunks_are_flagged():
    rng = np.random.default_rng(0)
    quiet_noise = (rng.standard_normal(20 * SAMPLE_RATE) * 1e-4).astype(np.float32)  # ~ -80 dBFS
    signal = np.concatenate([tone(20), np.zeros(20 * SAMPLE_RATE, np.float32), quiet_noise, tone(20)])
    chunks = make_chunks(signal)
    assert [c.silent for c in chunks] == [False, True, True, False]
    assert chunks[1].rms_dbfs == float("-inf")
    assert chunks[0].rms_dbfs == pytest.approx(20 * np.log10(0.3 / np.sqrt(2)), abs=0.1)


def test_silence_threshold_is_configurable():
    signal = tone(20, amplitude=0.01)  # about -43 dBFS
    assert make_chunks(signal, silence_dbfs=-45)[0].silent is False
    assert make_chunks(signal, silence_dbfs=-40)[0].silent is True


def test_chunk_times():
    chunks = make_chunks(tone(44))
    assert [(c.start, c.end) for c in chunks] == [(0, 20), (20, 44)]
    assert chunks[1].duration == 24


# --- ffmpeg conversion (local only, skipped without ffmpeg) -------------------

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_to_wav_resamples_downmixes_and_trims(tmp_path):
    src = tmp_path / "stereo44k.wav"
    rate = 44_100
    t = np.arange(rate * 4) / rate
    stereo = (np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1) * 0.5 * 32767).astype("<i2")
    with wave.open(str(src), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(stereo.tobytes())

    out = audio.to_wav(src, tmp_path / "out.wav", start=1.0, end=3.5)
    samples = audio.load_wav(out)
    assert len(samples) == pytest.approx(2.5 * SAMPLE_RATE, abs=SAMPLE_RATE * 0.05)
    assert samples.dtype == np.float32


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_to_wav_reports_unreadable_file(tmp_path):
    bad = tmp_path / "not_audio.mp3"
    bad.write_text("hello")
    with pytest.raises(audio.AudioError, match="ffmpeg could not read"):
        audio.to_wav(bad, tmp_path / "out.wav")


def test_require_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    with pytest.raises(audio.AudioError, match="ffmpeg was not found"):
        audio.require_ffmpeg()


# --- silence trimming inside chunks -------------------------------------------

def test_speech_mask_marks_tone_and_pads_edges():
    gap = np.zeros(2 * SAMPLE_RATE, np.float32)
    signal = np.concatenate([gap, tone(1), gap])
    mask = audio.speech_mask(signal)
    assert mask.shape == signal.shape and mask.dtype == bool
    kept = mask.sum() / SAMPLE_RATE
    # 1 s of tone plus up to SPEECH_PAD_SECONDS (+ one frame) either side.
    assert 1.0 + 2 * audio.SPEECH_PAD_SECONDS <= kept <= 1.0 + 2 * (audio.SPEECH_PAD_SECONDS + audio.FRAME_SECONDS)
    assert not mask[: SAMPLE_RATE].any() and not mask[-SAMPLE_RATE:].any()
    assert mask[2 * SAMPLE_RATE : 3 * SAMPLE_RATE].all()


def test_speech_mask_edge_cases():
    assert audio.speech_mask(np.zeros(0, np.float32)).size == 0
    assert not audio.speech_mask(np.zeros(SAMPLE_RATE, np.float32)).any()
    assert audio.speech_mask(tone(0.5)).all()


def test_chunk_speech_drops_inner_silence():
    signal = np.concatenate([tone(4), np.zeros(12 * SAMPLE_RATE, np.float32), tone(4)])
    (chunk,) = make_chunks(signal)
    assert not chunk.silent
    assert chunk.speech_seconds == pytest.approx(8 + 2 * audio.SPEECH_PAD_SECONDS, abs=0.1)
    speech = chunk.speech(signal)
    assert len(speech) == int(chunk.speech_seconds * SAMPLE_RATE)
    assert audio.rms_dbfs(speech) > audio.rms_dbfs(signal)


def test_chunk_speech_uses_chunk_offsets():
    signal = np.concatenate([np.zeros(20 * SAMPLE_RATE, np.float32), tone(20, amplitude=0.5)])
    chunks = make_chunks(signal)
    assert chunks[0].silent and not chunks[1].silent
    assert np.array_equal(chunks[1].speech(signal), signal[20 * SAMPLE_RATE :])


def test_chunk_with_too_little_speech_is_skipped():
    # A loud 0.3 s click lifts the chunk RMS above the threshold but is not speech.
    signal = np.concatenate([tone(0.3, amplitude=0.9), np.zeros(19 * SAMPLE_RATE, np.float32)])
    (chunk,) = make_chunks(signal)
    assert chunk.rms_dbfs > audio.SILENCE_DBFS
    assert chunk.speech_seconds < audio.MIN_SPEECH_SECONDS
    assert chunk.silent
