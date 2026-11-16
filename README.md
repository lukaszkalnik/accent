# accent

A small CLI that guesses the English accent of a speaker in an audio/video file or a
YouTube video, using the [CommonAccent](https://arxiv.org/abs/2305.18283) models from
SpeechBrain.

```
$ accent "https://www.youtube.com/watch?v=QXX-uwQ-VGw"
Video:    Anthropic's Fable 5.2 Is Shockingly Good at Research Math...
Channel:  Easy Riders
Duration: 11:40
Analysed: 0:00-10:00 (first 10 minutes only; use --full for the whole video)
Model:    ecapa (mps)

Top accents (10:00 of audio in 30 chunk(s), 0 silent skipped):
  1. English (England)  100.0%
  2. American (US)        0.0%
  3. New Zealand          0.0%

Note: this is a statistical guess over a fixed set of 16 accent labels; mixed, regional or non-native accents are often misclassified.
```

## Install

Needs Python 3.11+ and ffmpeg on `PATH` (`brew install ffmpeg` on macOS,
`winget install Gyan.FFmpeg` on Windows, `sudo apt install ffmpeg` on Debian/Ubuntu).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

Dependencies are pinned in `pyproject.toml`, except yt-dlp: YouTube changes often and old
yt-dlp releases stop working. If YouTube downloads start failing, run `pip install -U yt-dlp`.

yt-dlp needs a JavaScript runtime (Deno, Node.js, Bun or QuickJS) on `PATH` to
download from YouTube. The tool uses whichever one it finds.

The first run downloads the model to `~/.cache/accent-cli/models` (ECAPA is about 80 MB,
XLSR about 2.5 GB). Set `ACCENT_MODEL_DIR` to put it somewhere else.

## Usage

```bash
accent interview.m4a                          # any format ffmpeg reads: mp3, m4a, wav, mp4, webm...
accent talk.mp4 --start 1:30 --end 4:00       # only part of the file (seconds or mm:ss / hh:mm:ss)
accent talk.mp4 -v                            # add a per-chunk table
accent talk.mp4 --json                        # machine-readable output
accent talk.mp4 --model xlsr                  # larger wav2vec2 model
accent "https://youtu.be/QXX-uwQ-VGw"         # YouTube: first 10 minutes by default
accent "https://youtu.be/QXX-uwQ-VGw" --full  # the whole video
accent "https://www.youtube.com/watch?v=QXX-uwQ-VGw" --start 5:00 --end 6:00   # only that section is downloaded
```

| Option | Meaning |
| --- | --- |
| `--model ecapa\|xlsr` | `ecapa` (default): ECAPA-TDNN, small and fast on CPU. `xlsr`: wav2vec2 XLSR-53, much larger. |
| `--start`, `--end` | Analyse only this range. For YouTube only this section is downloaded. |
| `--full` | YouTube: no 10-minute cap. |
| `--no-cache` | YouTube: download again instead of reusing `.cache/youtube/`. |
| `-v`, `--verbose` | Print a per-chunk table. |
| `--json` | JSON on stdout (the note goes to stderr and is also in the JSON). |
| `--top N` | Number of accents to show (default 3). |
| `--device auto\|cpu\|mps\|cuda` | `auto` picks CUDA, then Apple MPS, then CPU. |
| `--ipv4` | Force IPv4 (or set `ACCENT_IPV4=1`). Use this if downloads hang or crawl on a network with broken IPv6. |

### How it works

1. The input is converted to 16 kHz mono WAV with ffmpeg (YouTube: yt-dlp fetches the best
   audio-only stream, never video, cut to the requested section).
2. The audio is split into 20-second chunks. A tail shorter than 5 s is merged into the
   previous chunk.
3. Chunks whose RMS level is below -45 dBFS are skipped as silence.
4. Each remaining chunk is classified. The final result is the average of the chunk
   probabilities, weighted by chunk duration.

### YouTube

- Accepted: `youtube.com/watch?v=…`, `youtu.be/…`, `/shorts/…`, `/live/…`, `/embed/…`,
  with or without extra query parameters (`&t=`, `&si=`, `&list=`…).
- Rejected with a message: playlists, live streams and upcoming premieres.
- Downloads are cached in `.cache/youtube/` (in the current directory), keyed by video ID and time range.
- Private, age-restricted, region-blocked or removed videos, and network failures, each
  produce one `error:` line and exit code 1.

## Models

| `--model` | Hugging Face ID | Notes |
| --- | --- | --- |
| `ecapa` | [`Jzuluaga/accent-id-commonaccent_ecapa`](https://huggingface.co/Jzuluaga/accent-id-commonaccent_ecapa) | ECAPA-TDNN, 87 % test accuracy per the model card |
| `xlsr` | [`Jzuluaga/accent-id-commonaccent_xlsr-en-english`](https://huggingface.co/Jzuluaga/accent-id-commonaccent_xlsr-en-english) | wav2vec2 XLSR-53 + custom SpeechBrain interface |

Both use the same 16 labels: african, australia, bermuda, canada, england, hongkong, indian,
ireland, malaysia, newzealand, philippines, scotland, singapore, southatlandtic (sic),
us, wales.

The ECAPA head outputs cosine similarities, not probabilities. The tool converts them with
`softmax(30 × cosine)`, the scale of the additive-angular-margin loss the model was trained
with. Expect very confident numbers: they reflect how the model was trained, not real
certainty.

## Tests

```bash
pytest                       # unit tests: no network, no model download
pytest -m integration        # real model on a sample clip + a short public YouTube video
```

## Windows

The code avoids POSIX-only calls. On Windows SpeechBrain copies model files instead of
symlinking them, because symlinks need admin rights or Developer Mode.
