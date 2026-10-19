#!/usr/bin/env python3
"""Local web app: drag a file (or paste a YouTube URL), pick stems, download the backing track."""

from __future__ import annotations

import io
import json
import threading
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import quote

from flask import Flask, Response, jsonify, render_template, request, send_file

import core
import tempo

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB

OUTPUT = core.ROOT / "output"


class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.status = "running"
        self.messages: list[dict] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.done = False
        self.started = time.time()
        self.lock = threading.Lock()

    def _emit(self, **fields) -> None:
        payload = {"elapsed": round(time.time() - self.started), **fields}
        self.messages.append(payload)

    def update(self, stage=None, detail=None, percent=None):
        with self.lock:
            self._emit(stage=stage, detail=detail, percent=percent, status="running")

    def finish(self, result: dict) -> None:
        with self.lock:
            self.result = result
            self.status = "done"
            self.done = True
            self._emit(stage="done", detail="Finished", status="done")

    def fail(self, error: str) -> None:
        with self.lock:
            self.error = error
            self.status = "error"
            self.done = True
            self._emit(stage="error", detail=error, status="error")


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def run_job(job: Job, source: str, keep, model: str, fmt: str, work: Path) -> None:
    try:
        result = core.run(
            source,
            keep=keep,
            model=model,
            fmt=fmt,
            all_stems=True,
            work=work,
            on_stage=lambda stage, detail="", percent=None: job.update(stage, detail, percent),
        )
        job.finish(result)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        job.fail(str(exc))


@app.get("/")
def index():
    return render_template("index.html", models=core.MODELS, labels=core.STEM_LABELS)


def _ordered_stems(stems: dict) -> dict:
    order = list(core.STEM_LABELS)
    return {
        name: stems[name]
        for name in sorted(stems, key=lambda s: (order.index(s) if s in order else len(order), s))
    }


def _entry_meta(entry_dir: Path) -> dict | None:
    meta_path = entry_dir / "job.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = None
        if isinstance(meta, dict):
            return meta

    # Fallback for tracks created before history metadata existed.
    stems_dir = entry_dir / "stems"
    stems = {
        path.stem: path.name
        for path in sorted(stems_dir.glob("*.wav"))
    } if stems_dir.is_dir() else {}
    mixes = sorted(entry_dir.glob("* - backing.*"))
    if not stems and not mixes:
        return None
    return {
        "name": entry_dir.name,
        "created": entry_dir.stat().st_mtime,
        "format": mixes[0].suffix.lstrip(".").lower() if mixes else None,
        "selected": list(stems),
        "mix": mixes[0].name if mixes else None,
        "stems": stems,
    }


def _entry_payload(entry_id: str, meta: dict) -> dict:
    selected = set(meta.get("selected", []))
    stems = _ordered_stems(meta.get("stems", {}))
    base = f"/history/{quote(entry_id)}"
    tracks = [
        {
            "name": stem,
            "label": core.stem_label(Path(stem)),
            "kept": stem in selected,
            "url": f"{base}/audio/{quote(stem)}",
        }
        for stem in stems
    ]
    return {
        "id": entry_id,
        "name": meta.get("name", entry_id),
        "created": meta.get("created", 0),
        "model": meta.get("model"),
        "format": meta.get("format"),
        "selected": meta.get("selected", []),
        "bpm": meta.get("bpm"),
        "backing": f"{base}/backing" if meta.get("mix") else None,
        "stems": f"{base}/stems.zip" if tracks else None,
        "tracks": tracks,
    }


def _load_entry(entry_id: str):
    root = OUTPUT.resolve()
    entry_dir = (OUTPUT / entry_id).resolve()
    if entry_dir.parent != root or not entry_dir.is_dir():
        return None, None
    meta = _entry_meta(entry_dir)
    if meta is None:
        return None, None
    return entry_dir, meta


def _safe_child(base: Path, name: str) -> Path | None:
    if not name:
        return None
    path = (base / name).resolve()
    if path.parent != base.resolve() or not path.is_file():
        return None
    return path


@app.get("/history")
def history():
    entries = []
    if OUTPUT.is_dir():
        for entry_dir in OUTPUT.iterdir():
            if not entry_dir.is_dir():
                continue
            meta = _entry_meta(entry_dir)
            if meta is None:
                continue
            entries.append(_entry_payload(entry_dir.name, meta))
    entries.sort(key=lambda entry: entry["created"], reverse=True)
    return jsonify(entries=entries[:40])


@app.get("/history/<entry_id>/audio/<stem_name>")
def history_audio(entry_id: str, stem_name: str):
    entry_dir, meta = _load_entry(entry_id)
    if entry_dir is None:
        return jsonify(error="Unknown entry"), 404
    filename = meta.get("stems", {}).get(stem_name)
    path = _safe_child(entry_dir / "stems", filename) if filename else None
    if path is None:
        return jsonify(error="Unknown stem"), 404
    return send_file(path, mimetype="audio/wav", conditional=True)


@app.get("/history/<entry_id>/bpm")
def history_bpm(entry_id: str):
    entry_dir, meta = _load_entry(entry_id)
    if entry_dir is None:
        return jsonify(error="Unknown entry"), 404
    if meta.get("bpm"):
        return jsonify(bpm=meta["bpm"])

    stems = meta.get("stems", {})
    filename = stems.get("drums") or next(iter(stems.values()), None)
    path = _safe_child(entry_dir / "stems", filename) if filename else None
    if path is None:
        return jsonify(error="No stems to analyze"), 404
    try:
        value = tempo.detect_bpm(path)
    except Exception as exc:  # noqa: BLE001 - report detection failure to the UI
        return jsonify(error=str(exc)), 422
    if value is None:
        return jsonify(error="Could not detect a tempo"), 422

    meta["bpm"] = value
    try:
        (entry_dir / "job.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:
        pass
    return jsonify(bpm=value)


@app.get("/history/<entry_id>/backing")
def history_backing(entry_id: str):
    entry_dir, meta = _load_entry(entry_id)
    path = _safe_child(entry_dir, meta.get("mix", "")) if meta else None
    if path is None:
        return jsonify(error="Not found"), 404
    return send_file(path, as_attachment=True, download_name=path.name)


@app.get("/history/<entry_id>/stems.zip")
def history_stems(entry_id: str):
    entry_dir, meta = _load_entry(entry_id)
    if entry_dir is None:
        return jsonify(error="Unknown entry"), 404
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in meta.get("stems", {}).values():
            path = _safe_child(entry_dir / "stems", name)
            if path is not None:
                archive.write(path, path.name)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{meta.get('name', entry_id)} - stems.zip",
    )


@app.post("/jobs")
def create_job():
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id)

    model = request.form.get("model", core.DEFAULT_MODEL)
    if model not in core.MODELS:
        return jsonify(error="Unknown model"), 400
    fmt = request.form.get("format", "mp3")
    if fmt not in core.FORMATS:
        return jsonify(error="Unknown format"), 400
    keep = [s for s in request.form.getlist("keep") if s in core.MODELS[model]] or None

    work = core.ROOT / "work" / job_id
    work.mkdir(parents=True, exist_ok=True)

    url = request.form.get("url", "").strip()
    upload = request.files.get("file")
    if upload and upload.filename:
        safe = core.safe_name(Path(upload.filename).name)
        source = work / "uploads" / safe
        source.parent.mkdir(parents=True, exist_ok=True)
        upload.save(source)
        source = str(source)
    elif url:
        source = url
    else:
        return jsonify(error="Provide a file or a URL"), 400

    with JOBS_LOCK:
        JOBS[job_id] = job
    threading.Thread(
        target=run_job, args=(job, source, keep, model, fmt, work), daemon=True
    ).start()
    return jsonify(id=job_id)


@app.get("/jobs/<job_id>/events")
def job_events(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify(error="Unknown job"), 404

    def stream():
        idx = 0
        while True:
            with job.lock:
                pending = job.messages[idx:]
                idx = len(job.messages)
                done = job.done
            for message in pending:
                if message.get("status") in {"done", "error"}:
                    message = {**message, "result": _result_urls(job)}
                yield f"data: {json.dumps(message)}\n\n"
            if done:
                break
            time.sleep(0.3)

    return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})


def _result_urls(job: Job) -> dict | None:
    if job.result is None:
        return None
    selected = set(job.result["selected"])
    tracks = [
        {
            "name": stem.stem,
            "label": core.stem_label(stem),
            "kept": stem.stem in selected,
            "url": f"/jobs/{job.id}/audio/{stem.stem}",
        }
        for stem in job.result["stems"]
    ]
    return {
        "name": job.result["name"],
        "selected": job.result["selected"],
        "bpm": job.result.get("bpm"),
        "backing": f"/jobs/{job.id}/download/backing",
        "stems": f"/jobs/{job.id}/download/stems.zip" if job.result["stems"] else None,
        "tracks": tracks,
    }


@app.get("/jobs/<job_id>/audio/<stem_name>")
def job_audio(job_id: str, stem_name: str):
    job = JOBS.get(job_id)
    if job is None or job.result is None:
        return jsonify(error="Not ready"), 404
    stem = next((s for s in job.result["stems"] if s.stem == stem_name), None)
    if stem is None:
        return jsonify(error="Unknown stem"), 404
    return send_file(stem, mimetype="audio/wav", conditional=True)


@app.get("/jobs/<job_id>/download/backing")
def download_backing(job_id: str):
    job = JOBS.get(job_id)
    if job is None or job.result is None:
        return jsonify(error="Not ready"), 404
    mix = job.result["mix"]
    return send_file(mix, as_attachment=True, download_name=mix.name)


@app.get("/jobs/<job_id>/download/stems.zip")
def download_stems(job_id: str):
    job = JOBS.get(job_id)
    if job is None or job.result is None:
        return jsonify(error="Not ready"), 404
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for stem in job.result["stems"]:
            archive.write(stem, stem.name)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{job.result['name']} - stems.zip",
    )


if __name__ == "__main__":
    print("\033[1;36m::\033[0m Open http://127.0.0.1:8756 in your browser")
    app.run(host="127.0.0.1", port=8756, threaded=True)
