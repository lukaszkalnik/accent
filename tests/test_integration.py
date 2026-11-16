"""Integration tests: real model and real YouTube download. Need network; not run by default.

    pytest -m integration          # set ACCENT_IPV4=1 if your network's IPv6 is broken
"""

import json

import pytest

from accent import cli

pytestmark = pytest.mark.integration


def run_json(capsys, *args):
    code = cli.main(["--json", *args])
    out = capsys.readouterr()
    assert code == 0, out.err
    return json.loads(out.out), out.err


def test_ecapa_on_model_card_sample(capsys):
    from huggingface_hub import hf_hub_download

    sample = hf_hub_download("Jzuluaga/accent-id-commonaccent_ecapa", "data/england_1.wav")
    result, err = run_json(capsys, sample)
    assert result["top"][0]["label"] == "england"
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert "statistical guess" in err


def test_short_public_video(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep the download cache out of the repo
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo", 19 s
    result, _ = run_json(capsys, "--no-cache", url)
    assert result["source"]["video_id"] == "jNQXAC9IVRw"
    assert result["source"]["title"]
    assert result["source"]["duration"] < 30
    assert result["range"]["capped"] is False
    assert len(result["top"]) == 3
    assert list((tmp_path / ".cache" / "youtube").glob("jNQXAC9IVRw_0-end.*"))

    # Second run comes from the cache.
    cached, _ = run_json(capsys, url)
    assert cached["source"]["from_cache"] is True
    assert cached["top"] == result["top"]
