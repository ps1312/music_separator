#!/usr/bin/env python3
"""Shared pipeline: download, Demucs separation, stem selection and mixing."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import tempo

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "htdemucs"

STEM_LABELS = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "guitar": "Guitar",
    "piano": "Piano",
    "other": "Other",
}

MODELS = {
    "htdemucs": ["vocals", "drums", "bass", "other"],
}

FORMATS = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
    "wav": ["-c:a", "pcm_s16le"],
    "flac": ["-c:a", "flac"],
}


class SeparationError(Exception):
    pass


def require(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise SeparationError(f"`{cmd}` is not installed or not on your PATH.")


def is_url(source: str) -> bool:
    return bool(re.match(r"^https?://", source, re.I))


def download(url: str, work: Path, on_stage=None) -> Path:
    require("yt-dlp")
    dest = work / "downloads"
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format",
            "wav",
            "--no-playlist",
            "--print",
            "after_move:filepath",
            "-o",
            str(dest / "%(title)s.%(ext)s"),
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SeparationError(f"yt-dlp failed:\n{result.stderr.strip()[-500:]}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise SeparationError("yt-dlp did not report an output file.")
    path = Path(lines[-1])
    if not path.exists():
        raise SeparationError(f"Downloaded file not found: {path}")
    return path


def separate(audio: Path, model: str, work: Path, device=None, jobs: int = 0, on_percent=None):
    require("ffmpeg")
    out_root = work / "separated"
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(out_root), "-j", str(jobs)]
    if device:
        cmd += ["-d", device]
    cmd.append(str(audio))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    buffer = ""
    read = lambda: proc.stdout.read(256)
    for chunk in iter(read, ""):
        buffer += chunk
        parts = re.split(r"[\r\n]", buffer)
        buffer = parts.pop()
        for line in parts:
            line = line.strip()
            if not line:
                continue
            match = re.search(r"(\d+)%", line)
            if match and on_percent:
                on_percent(int(match.group(1)))
    proc.wait()
    if proc.returncode != 0:
        raise SeparationError("Demucs failed to separate the track.")

    track_dirs = sorted(out_root.joinpath(model).glob("*"), key=lambda p: p.stat().st_mtime)
    if not track_dirs:
        raise SeparationError("Could not find Demucs output.")
    track_dir = track_dirs[-1]
    stems = sorted(track_dir.glob("*.wav"))
    if not stems:
        raise SeparationError("Demucs produced no stems.")
    return track_dir, stems


def order_stems(stems):
    known = list(STEM_LABELS)
    return sorted(stems, key=lambda p: (known.index(p.stem) if p.stem in known else len(known), p.stem))


def parse_selection(raw: str, stems) -> list[str]:
    """Parse a CLI selection string into a list of stem names."""
    names = [s.stem for s in stems]
    raw = (raw or "").strip().lower()
    if not raw or raw in {"default", "d"}:
        return [n for n in names if n != "guitar"] or names
    if raw == "all":
        return names
    if raw == "none":
        return []
    chosen: list[str] = []
    for token in re.split(r"[,\s]+", raw):
        if not token:
            continue
        if "-" in token and token.replace("-", "").isdigit():
            start, end = (int(x) for x in token.split("-"))
            chosen.extend(names[i - 1] for i in range(start, end + 1) if 1 <= i <= len(names))
        elif token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(names):
                chosen.append(names[idx - 1])
        elif token in names:
            chosen.append(token)
        else:
            raise SeparationError(f"Unknown stem: {token}")
    deduped: list[str] = []
    for name in chosen:
        if name not in deduped:
            deduped.append(name)
    return deduped


def select_stems(stems, keep=None, exclude=None):
    names = [s.stem for s in stems]
    if keep:
        chosen = [n for n in names if n in set(keep)]
    else:
        chosen = [n for n in names if n != "guitar"] or names
    if exclude:
        excluded = set(exclude)
        chosen = [n for n in chosen if n not in excluded]
    selected = [s for s in stems if s.stem in chosen]
    if not selected:
        raise SeparationError("No stems selected.")
    return selected


def stem_label(path: Path) -> str:
    return STEM_LABELS.get(path.stem, path.stem.title())


def mix(stems, out_path: Path, fmt: str) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for stem in stems:
        cmd += ["-i", str(stem)]
    if len(stems) > 1:
        chain = f"amix=inputs={len(stems)}:duration=longest:normalize=0,alimiter=limit=0.95[out]"
        cmd += ["-filter_complex", chain, "-map", "[out]"]
    else:
        cmd += ["-map", "0:a:0"]
    cmd += FORMATS[fmt] + ["-vn", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SeparationError(f"ffmpeg failed to mix the stems:\n{result.stderr.strip()[-500:]}")


def safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "-", name).strip(" .")
    return name or "track"


def guess_track_name(audio: Path, track_dir: Path) -> str:
    if audio.name.startswith(track_dir.name):
        return track_dir.name
    return audio.stem


def run(
    source: str,
    *,
    keep=None,
    exclude=None,
    model: str = DEFAULT_MODEL,
    device=None,
    jobs: int = 0,
    fmt: str = "mp3",
    out=None,
    no_stems: bool = False,
    all_stems: bool = False,
    clean_work: bool = True,
    work: Path | None = None,
    on_stage=None,
) -> dict:
    """Run the full pipeline. `source` is a URL or a local file path."""
    emit = on_stage or (lambda *a, **k: None)
    work = Path(work) if work else ROOT / "work"
    work.mkdir(parents=True, exist_ok=True)

    if is_url(source):
        emit("download", "Downloading audio from YouTube")
        audio = download(source, work)
    else:
        audio = Path(source).expanduser()
        if not audio.exists():
            raise SeparationError(f"File not found: {audio}")

    emit("separate", f"Separating stems with {model}", percent=0)
    track_dir, stems = separate(
        audio, model, work, device, jobs, on_percent=lambda p: emit("separate", percent=p)
    )
    stems = order_stems(stems)
    selected = select_stems(stems, keep, exclude)

    emit("bpm", "Detecting tempo")
    detected_bpm = None
    try:
        drums = next((stem for stem in stems if stem.stem == "drums"), None)
        detected_bpm = tempo.detect_bpm(drums or selected[0])
    except Exception:
        detected_bpm = None

    name = safe_name(guess_track_name(audio, track_dir))
    out_dir = (Path(out) if out else ROOT / "output") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    emit("mix", "Mixing the selected stems")
    mix_path = out_dir / f"{name} - backing.{fmt}"
    mix(selected, mix_path, fmt)

    stem_paths = []
    if not no_stems:
        stems_dir = out_dir / "stems"
        stems_dir.mkdir(exist_ok=True)
        for stem in stems if all_stems else selected:
            dest = stems_dir / f"{stem_label(stem).lower()}.wav"
            shutil.copy2(stem, dest)
            stem_paths.append(dest)

    metadata = {
        "name": name,
        "created": time.time(),
        "model": model,
        "format": fmt,
        "selected": [s.stem for s in selected],
        "mix": mix_path.name,
        "stems": {p.stem: p.name for p in stem_paths},
        "bpm": detected_bpm,
    }
    (out_dir / "job.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if clean_work:
        shutil.rmtree(work, ignore_errors=True)

    result = {
        "name": name,
        "dir": out_dir,
        "mix": mix_path,
        "stems": stem_paths,
        "selected": [s.stem for s in selected],
        "model": model,
        "format": fmt,
        "bpm": detected_bpm,
    }
    emit("done", "Finished")
    return result
