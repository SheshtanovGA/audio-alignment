"""Web UI for opera_align — same commands, options, and defaults as the CLI."""
import argparse
import io
import os
import sys
import traceback
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import defaults as D
from .cli import cmd_align, cmd_map_subtitles, cmd_pipeline, cmd_plot, cmd_warp

_PKG_DIR = Path(__file__).resolve().parent
UPLOAD_ROOT = Path(os.environ.get("OPERA_ALIGN_UPLOAD_DIR", "uploads"))
WORK_ROOT = Path(os.environ.get("OPERA_ALIGN_WORK_DIR", ".")).resolve()

COMMANDS = ("align", "pipeline", "map-subtitles", "plot", "warp-video")

FILE_FIELDS = {
    "align": ("ref_wav", "stream_wav", "subtitles"),
    "pipeline": ("ref_wav", "stream_wav", "video", "subtitles"),
    "map-subtitles": ("subtitles_csv",),
    "plot": ("subtitles_csv",),
    "warp-video": ("input_video", "curve_csv"),
}

PATH_OVERRIDE_FIELDS = tuple(D.ALIGNMENT_ARTIFACT_NAMES)


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _request():
    from flask import request
    return request


def _form_bool(name: str) -> bool:
    return _request().form.get(name) == "on"


def _save_upload(field: str, dest_dir: Path) -> Optional[str]:
    from werkzeug.utils import secure_filename

    f = _request().files.get(field)
    if f is None or not f.filename:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = secure_filename(f.filename)
    path = dest_dir / name
    f.save(path)
    return str(path.resolve())


def _collect_uploads(cmd: str, job_dir: Path) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for field in FILE_FIELDS.get(cmd, ()):
        saved = _save_upload(field, job_dir / "inputs")
        if saved:
            paths[field] = saved
    for field in PATH_OVERRIDE_FIELDS:
        saved = _save_upload(field, job_dir / "artifacts_override")
        if saved:
            paths[field] = saved
    return paths


def _build_namespace(cmd: str, uploads: Dict[str, str]) -> argparse.Namespace:
    """Map form fields to an argparse.Namespace matching the CLI."""
    ns = argparse.Namespace(cmd=cmd)

    if cmd == "align":
        form = _request().form
        ns.ref_wav = uploads.get("ref_wav") or form.get("ref_wav_path", "").strip()
        ns.stream_wav = uploads.get("stream_wav") or form.get("stream_wav_path", "").strip()
        ns.subtitles = uploads.get("subtitles") or form.get("subtitles_path") or None
        if ns.subtitles == "":
            ns.subtitles = None
        ns.out_prefix = form.get("out_prefix", D.ALIGN_OUT_PREFIX)
        ns.out_subtitles = form.get("out_subtitles", D.ALIGN_OUT_SUBTITLES)
        ns.artifacts_dir = D.ARTIFACTS_DIR
        ns.sr = int(form.get("sr", D.SR))
        ns.feature = form.get("feature", D.FEATURE)
        ns.embedding_size = int(form.get("embedding_size", D.EMBEDDING_SIZE))
        ns.hop_size = float(form.get("hop_size", D.HOP_SIZE))
        ns.hop_length = _optional_int(form.get("hop_length"))
        ns.n_mfcc = int(form.get("n_mfcc", D.N_MFCC))
        ns.chroma_type = form.get("chroma_type", D.CHROMA_TYPE)
        ns.backend = form.get("backend", D.BACKEND)
        return ns

    if cmd == "pipeline":
        ns.ref_wav = uploads.get("ref_wav") or request.form.get("ref_wav_path", "").strip()
        ns.stream_wav = uploads.get("stream_wav") or request.form.get("stream_wav_path", "").strip()
        ns.video = uploads.get("video") or request.form.get("video_path", "").strip()
        ns.session = request.form.get("session") or None
        if ns.session == "":
            ns.session = None
        ns.output_dir = request.form.get("output_dir", D.PIPELINE_OUTPUT_DIR)
        ns.artifacts_dir = request.form.get("artifacts_dir", D.PIPELINE_ARTIFACTS_DIR)
        ns.plot_name = request.form.get("plot_name", D.PIPELINE_PLOT_NAME)
        ns.video_name = request.form.get("video_name", D.PIPELINE_VIDEO_NAME)
        ns.subtitles = uploads.get("subtitles") or request.form.get("subtitles_path") or None
        if ns.subtitles == "":
            ns.subtitles = None
        ns.no_audio = _form_bool("no_audio")
        ns.sr = int(request.form.get("sr", D.SR))
        ns.feature = request.form.get("feature", D.FEATURE)
        ns.embedding_size = int(request.form.get("embedding_size", D.EMBEDDING_SIZE))
        ns.hop_size = float(request.form.get("hop_size", D.HOP_SIZE))
        ns.hop_length = _optional_int(request.form.get("hop_length"))
        ns.n_mfcc = int(request.form.get("n_mfcc", D.N_MFCC))
        ns.chroma_type = request.form.get("chroma_type", D.CHROMA_TYPE)
        ns.backend = request.form.get("backend", D.BACKEND)
        return ns

    ns.session = request.form.get("session") or None
    if ns.session == "":
        ns.session = None
    ns.artifacts_dir = request.form.get("artifacts_dir", D.ARTIFACTS_DIR)
    for name in PATH_OVERRIDE_FIELDS:
        setattr(
            ns,
            name,
            uploads.get(name) or request.form.get(f"{name}_path") or None,
        )
        val = getattr(ns, name)
        if val == "":
            setattr(ns, name, None)

    if cmd == "map-subtitles":
        ns.subtitles_csv = uploads.get("subtitles_csv") or request.form.get("subtitles_csv_path", "").strip()
        ns.output_csv = request.form.get("output_csv", D.MAP_OUTPUT_CSV)
        return ns

    if cmd == "plot":
        ns.subtitles_csv = uploads.get("subtitles_csv") or request.form.get("subtitles_csv_path") or None
        if ns.subtitles_csv == "":
            ns.subtitles_csv = None
        ns.output_png = request.form.get("output_png", D.PLOT_OUTPUT_PNG)
        return ns

    if cmd == "warp-video":
        ns.input_video = uploads.get("input_video") or request.form.get("input_video_path", "").strip()
        ns.output_video = request.form.get("output_video", "").strip()
        ns.curve_csv = uploads.get("curve_csv") or request.form.get("curve_csv_path") or None
        if ns.curve_csv == "":
            ns.curve_csv = None
        ns.no_audio = _form_bool("no_audio")
        return ns

    raise ValueError(f"Unknown command: {cmd}")


def _validate_namespace(cmd: str, ns: argparse.Namespace) -> Optional[str]:
    if cmd == "align":
        if not ns.ref_wav or not ns.stream_wav:
            return "Reference and stream WAV are required (upload or path)."
        return None
    if cmd == "pipeline":
        if not ns.ref_wav or not ns.stream_wav or not ns.video:
            return "Reference WAV, stream WAV, and video are required."
        return None
    if cmd == "map-subtitles":
        if not ns.subtitles_csv:
            return "Subtitles CSV is required."
        if not ns.session and not all(getattr(ns, n) for n in PATH_OVERRIDE_FIELDS):
            return "Provide session name or all four alignment artifact paths."
        return None
    if cmd == "plot":
        if not ns.session and not all(getattr(ns, n) for n in PATH_OVERRIDE_FIELDS):
            return "Provide session name or all four alignment artifact paths."
        return None
    if cmd == "warp-video":
        if not ns.input_video or not ns.output_video:
            return "Input video and output video name are required."
        needs_artifacts = not ns.curve_csv
        if needs_artifacts and not ns.session and not all(getattr(ns, n) for n in PATH_OVERRIDE_FIELDS):
            return "Provide session, curve CSV, or all four alignment artifact paths."
        return None
    return f"Unknown command: {cmd}"


def _run_command(cmd: str, ns: argparse.Namespace) -> str:
    handlers = {
        "align": cmd_align,
        "pipeline": cmd_pipeline,
        "map-subtitles": cmd_map_subtitles,
        "plot": cmd_plot,
        "warp-video": cmd_warp,
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        handlers[cmd](ns)
    return buf.getvalue()


def _list_output_files() -> List[str]:
    out_dir = WORK_ROOT / "output"
    if not out_dir.is_dir():
        return []
    files = []
    for root, _, names in os.walk(out_dir):
        for name in names:
            full = Path(root) / name
            try:
                rel = full.relative_to(WORK_ROOT)
            except ValueError:
                rel = full
            files.append(str(rel).replace("\\", "/"))
    return sorted(files)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(_PKG_DIR / "templates"),
        static_folder=str(_PKG_DIR / "static"),
    )
    app.secret_key = os.environ.get("OPERA_ALIGN_SECRET_KEY", "dev-change-me")

    @app.route("/")
    def index():
        cmd = request.args.get("cmd", "pipeline")
        if cmd not in COMMANDS:
            cmd = "pipeline"
        return render_template("index.html", cmd=cmd, defaults=D, commands=COMMANDS)

    @app.route("/run", methods=["POST"])
    def run():
        cmd = request.form.get("cmd", "pipeline")
        if cmd not in COMMANDS:
            flash("Invalid command.", "error")
            return redirect(url_for("index"))

        job_id = request.form.get("job_id") or os.urandom(8).hex()
        job_dir = UPLOAD_ROOT / job_id
        uploads = _collect_uploads(cmd, job_dir)

        try:
            os.chdir(WORK_ROOT)
            ns = _build_namespace(cmd, uploads)
            err = _validate_namespace(cmd, ns)
            if err:
                flash(err, "error")
                return redirect(url_for("index", cmd=cmd))

            if cmd in ("plot", "map-subtitles", "warp-video"):
                from .cli import _alignment_paths_from_args

                needs_artifacts = cmd != "warp-video" or not ns.curve_csv
                if needs_artifacts:
                    _alignment_paths_from_args(ns)

            log = _run_command(cmd, ns)
            outputs = _list_output_files()
            return render_template(
                "result.html",
                cmd=cmd,
                log=log,
                outputs=outputs,
                job_id=job_id,
            )
        except Exception as e:
            log = traceback.format_exc()
            flash(str(e), "error")
            return render_template(
                "result.html",
                cmd=cmd,
                log=log,
                outputs=_list_output_files(),
                job_id=job_id,
                error=str(e),
            )

    @app.route("/files/<path:filepath>")
    def download_file(filepath):
        safe = Path(filepath)
        if ".." in safe.parts:
            return "Invalid path", 400
        directory = WORK_ROOT / safe.parent
        return send_from_directory(directory, safe.name, as_attachment=True)

    return app


def main():
    import argparse as ap

    parser = ap.ArgumentParser(description="opera_align web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    (WORK_ROOT / "output").mkdir(exist_ok=True)
    (WORK_ROOT / "artifacts").mkdir(exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK_ROOT)
    app = create_app()
    print(f"Working directory: {WORK_ROOT}")
    print(f"Open http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
