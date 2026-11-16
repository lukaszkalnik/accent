"""Command-line entry point: `accent <file or YouTube URL>`."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
from pathlib import Path

from . import __version__
from .audio import (
    SAMPLE_RATE,
    AudioError,
    format_seconds,
    load_wav,
    make_chunks,
    parse_time,
    require_ffmpeg,
    to_wav,
    validate_range,
)
from .classify import MODELS, AccentClassifier, ModelError, average_probabilities, display_name, top_k
from .youtube import DEFAULT_CAP_SECONDS, YouTubeError, fetch, looks_like_url

NOTE = (
    "Note: this is a statistical guess over a fixed set of 16 accent labels; "
    "mixed, regional or non-native accents are often misclassified."
)
CACHE_DIR = Path(".cache") / "youtube"


def _time_arg(value: str) -> float:
    try:
        return parse_time(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="accent",
        description="Guess the English accent of the speaker in an audio/video file or YouTube video.",
    )
    parser.add_argument("input", help="audio/video file (anything ffmpeg reads) or a YouTube video URL")
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="ecapa",
        help="ecapa: small and fast (default); xlsr: larger wav2vec2 model (~2.5 GB download)",
    )
    parser.add_argument("--start", type=_time_arg, metavar="TIME", help="analyse from here (seconds or mm:ss)")
    parser.add_argument("--end", type=_time_arg, metavar="TIME", help="analyse up to here (seconds or mm:ss)")
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"YouTube: analyse the whole video instead of the first {DEFAULT_CAP_SECONDS / 60:.0f} minutes",
    )
    parser.add_argument("--no-cache", action="store_true", help="YouTube: ignore .cache/ and download again")
    parser.add_argument("-v", "--verbose", action="store_true", help="also print a per-chunk table")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--top", type=int, default=3, metavar="N", help="number of accents to show (default 3)")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--ipv4",
        action="store_true",
        help="force IPv4 for downloads (use if downloads hang on a network with broken IPv6)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def force_ipv4() -> None:
    """Make every library in this process resolve hostnames to IPv4 only."""
    original = socket.getaddrinfo

    def ipv4_only(*args, **kwargs):
        results = original(*args, **kwargs)
        v4 = [r for r in results if r[0] == socket.AF_INET]
        return v4 or results

    socket.getaddrinfo = ipv4_only


def status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def analyse(args: argparse.Namespace) -> dict:
    require_ffmpeg()
    validate_range(args.start, args.end)
    if args.top < 1:
        raise ValueError("--top must be at least 1")

    result: dict = {"model": args.model}
    with tempfile.TemporaryDirectory(prefix="accent-") as tmp:
        wav_path = Path(tmp) / "audio.wav"

        if looks_like_url(args.input):
            def show_info(info, begin, finish, capped):
                if not args.json:
                    print(f"Video:    {info.title}")
                    print(f"Channel:  {info.channel}")
                    print(f"Duration: {format_seconds(info.duration) if info.duration else 'unknown'}")
                    span = f"{format_seconds(begin)}-{format_seconds(finish) if finish is not None else 'end'}"
                    cap_note = (
                        f" (first {DEFAULT_CAP_SECONDS / 60:.0f} minutes only; use --full for the whole video)"
                        if capped else ""
                    )
                    print(f"Analysed: {span}{cap_note}", flush=True)
                status("Fetching audio...")

            download = fetch(
                args.input,
                CACHE_DIR,
                start=args.start,
                end=args.end,
                full=args.full,
                use_cache=not args.no_cache,
                ipv4=args.ipv4,
                on_info=show_info,
            )
            if download.from_cache:
                status(f"Using cached audio: {download.path}")
            to_wav(download.path, wav_path)  # the download already covers just the section
            offset = download.start
            result["source"] = {
                "type": "youtube",
                "url": args.input,
                "video_id": download.info.id,
                "title": download.info.title,
                "channel": download.info.channel,
                "duration": download.info.duration,
                "from_cache": download.from_cache,
            }
            result["range"] = {"start": download.start, "end": download.end, "capped": download.capped}
        else:
            src = Path(args.input).expanduser()
            if not src.is_file():
                raise AudioError(f"file not found: {args.input}")
            to_wav(src, wav_path, start=args.start, end=args.end)
            offset = args.start or 0.0
            result["source"] = {"type": "file", "path": str(src)}
            result["range"] = {"start": offset, "end": args.end, "capped": False}

        samples = load_wav(wav_path)

    if samples.size < SAMPLE_RATE // 2:
        raise AudioError("no audio in the selected range")

    chunks = make_chunks(samples)
    speech = [c for c in chunks if not c.silent]
    if not speech:
        raise AudioError("no speech found: every chunk is below the silence threshold")

    classifier = AccentClassifier(args.model, args.device)
    status(f"Loading {args.model} model on {classifier.device}...")
    classifier.load()
    status(f"Classifying {len(speech)} chunk(s)...")

    chunk_rows = []
    per_chunk = []
    for chunk in chunks:
        row = {
            "index": chunk.index + 1,
            "start": round(offset + chunk.start, 2),
            "end": round(offset + chunk.end, 2),
            "rms_dbfs": None if chunk.rms_dbfs == float("-inf") else round(chunk.rms_dbfs, 1),
            "silent": chunk.silent,
        }
        if not chunk.silent:
            a, b = int(chunk.start * SAMPLE_RATE), int(chunk.end * SAMPLE_RATE)
            probs = classifier.predict(samples[a:b])
            per_chunk.append(probs)
            label, p = top_k(probs, classifier.labels, 1)[0]
            row.update(top_label=label, top_probability=round(p, 4))
        chunk_rows.append(row)

    mean = average_probabilities(per_chunk, weights=[c.duration for c in speech])
    result.update(
        device=classifier.device,
        top=[
            {"label": label, "name": display_name(label), "probability": round(p, 4)}
            for label, p in top_k(mean, classifier.labels, args.top)
        ],
        probabilities={label: round(float(p), 4) for label, p in zip(classifier.labels, mean)},
        chunks=chunk_rows,
        summary={
            "chunks": len(chunks),
            "silent_chunks": len(chunks) - len(speech),
            "analysed_seconds": round(sum(c.duration for c in speech), 2),
        },
        note=NOTE,
    )
    return result


def render_text(result: dict, verbose: bool) -> str:
    summary = result["summary"]
    lines = [
        f"Model:    {result['model']} ({result['device']})",
        "",
        f"Top accents ({format_seconds(summary['analysed_seconds'])} of audio in "
        f"{summary['chunks'] - summary['silent_chunks']} chunk(s), {summary['silent_chunks']} silent skipped):",
    ]
    width = max(len(t["name"]) for t in result["top"])
    for rank, t in enumerate(result["top"], 1):
        lines.append(f"  {rank}. {t['name']:<{width}}  {t['probability'] * 100:5.1f}%")

    if verbose:
        lines += ["", f"  {'#':>3}  {'Time':<15}  {'Level':>7}  Top accent"]
        for row in result["chunks"]:
            span = f"{format_seconds(row['start'])}-{format_seconds(row['end'])}"
            level = "-inf dB" if row["rms_dbfs"] is None else f"{row['rms_dbfs']:.0f} dB"
            if row["silent"]:
                detail = "(silent, skipped)"
            else:
                detail = f"{display_name(row['top_label'])} {row['top_probability'] * 100:.0f}%"
            lines.append(f"  {row['index']:>3}  {span:<15}  {level:>7}  {detail}")

    lines += ["", result["note"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ipv4:
        force_ipv4()
    try:
        result = analyse(args)
    except (AudioError, YouTubeError, ModelError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # last resort: one readable line, no stack trace
        if os.environ.get("ACCENT_DEBUG"):
            raise
        print(f"error: unexpected {exc.__class__.__name__}: {exc} (set ACCENT_DEBUG=1 for details)", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(NOTE, file=sys.stderr)
    else:
        print(render_text(result, args.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())
