#!/usr/bin/env python3
"""Download a track from YouTube and split it into stems with Demucs.

Pick which stems to keep to build a custom backing track.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import core


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a YouTube track, split it with Demucs, and keep only the stems you want.",
        epilog=(
            "Examples:\n"
            "  ./separate 'https://youtu.be/...' --keep vocals,drums,bass,piano,other\n"
            "  ./separate song.mp3 --exclude guitar\n"
            "  ./separate 'https://youtu.be/...' --keep all --format wav"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", nargs="?", help="YouTube URL or local audio file")
    parser.add_argument("--keep", help="Comma-separated stems to keep (names or numbers)")
    parser.add_argument("--exclude", help="Comma-separated stems to drop after selection")
    parser.add_argument("--model", default=core.DEFAULT_MODEL, choices=sorted(core.MODELS))
    parser.add_argument("--device", help="Demucs device, e.g. cpu, cuda, mps")
    parser.add_argument("--jobs", type=int, default=0, help="Parallel jobs for Demucs (default: auto)")
    parser.add_argument("--format", choices=sorted(core.FORMATS), default="mp3")
    parser.add_argument("--out", type=Path, default=core.ROOT / "output")
    parser.add_argument("--no-stems", action="store_true", help="Only write the mixed track")
    parser.add_argument("--keep-work", action="store_true", help="Keep intermediate files in ./work")
    args = parser.parse_args()

    source = args.source or input("YouTube URL or audio file: ").strip()
    if not source:
        print("error: no source given.", file=sys.stderr)
        raise SystemExit(1)

    # Interactive stem picker only makes sense before separation when no --keep is given.
    keep = None
    if args.keep:
        stems = [Path(f"{n}.wav") for n in core.MODELS[args.model]]
        keep = core.parse_selection(args.keep, stems)
        print(f"\033[1;36m::\033[0m Keeping: {', '.join(keep)}")
    exclude = None
    if args.exclude:
        stems = [Path(f"{n}.wav") for n in core.MODELS[args.model]]
        exclude = core.parse_selection(args.exclude, stems)

    if not args.keep:
        names = core.MODELS[args.model]
        print("\nAvailable stems:")
        for i, name in enumerate(names, 1):
            print(f"  {i}. {core.STEM_LABELS[name]}")
        try:
            raw = input(f"\nStems to KEEP [all except guitar]: ").strip()
        except EOFError:
            raw = ""
        keep = core.parse_selection(raw, [Path(f"{n}.wav") for n in names])

    stages = {
        "download": "Downloading audio...",
        "separate": "Separating stems...",
        "bpm": "Detecting tempo...",
        "mix": "Mixing selected stems...",
    }
    last_percent = [-1]

    def on_stage(stage, detail="", percent=None):
        if percent is not None:
            if percent != last_percent[0]:
                last_percent[0] = percent
                print(f"\r\033[1;36m::\033[0m Separating... {percent}%", end="", flush=True)
            return
        print(f"\r\033[1;36m::\033[0m {stages.get(stage, detail or stage)}" + " " * 20)

    try:
        result = core.run(
            source,
            keep=keep,
            exclude=exclude,
            model=args.model,
            device=args.device,
            jobs=args.jobs,
            fmt=args.format,
            out=args.out,
            no_stems=args.no_stems,
            clean_work=not args.keep_work,
            on_stage=on_stage,
        )
    except core.SeparationError as exc:
        print(f"\n\033[1;31merror:\033[0m {exc}", file=sys.stderr)
        raise SystemExit(1)

    tempo_note = f" ({result['bpm']:.1f} BPM)" if result.get("bpm") else ""
    print(f"\n\nDone: {result['dir']}{tempo_note}")


if __name__ == "__main__":
    main()
