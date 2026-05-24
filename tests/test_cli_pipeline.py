import os
import shutil
from pathlib import Path

import pytest

from opera_align.cli import main


def _has_ffmpeg():
    return shutil.which("ffmpeg") is not None


def test_cli_align_plot_warp_pipeline(tmp_path, monkeypatch):
    pytest.importorskip("openl3")
    pytest.importorskip("moviepy")

    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "tests" / "sources"

    ref_wav = source_dir / "test_ref.wav"
    stream_wav = source_dir / "test_stream.wav"
    subtitles_csv = source_dir / "test_subtitles.csv"
    input_video = source_dir / "test_video.mp4"

    monkeypatch.chdir(tmp_path)

    # Create output directories that CLI expects
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "artifacts").mkdir(exist_ok=True)

    out_prefix = "testsession"
    main([
        "align",
        "--ref_wav",
        str(ref_wav),
        "--stream_wav",
        str(stream_wav),
        "--subtitles",
        str(subtitles_csv),
        "--out_prefix",
        out_prefix,
        "--out_subtitles",
        "mapped_subtitles.csv",
    ])

    artifact_dir = tmp_path / "artifacts" / out_prefix
    assert (artifact_dir / "ts_ref.npy").exists()
    assert (artifact_dir / "ts_stream.npy").exists()
    assert (artifact_dir / "path_ref.npy").exists()
    assert (artifact_dir / "path_stream.npy").exists()

    main([
        "plot",
        "--session",
        out_prefix,
        "--output_png",
        "test_alignment.png",
    ])

    output_png = tmp_path / "output" / "test_alignment.png"
    assert output_png.exists()

    main([
        "warp-video",
        "--input_video",
        str(input_video),
        "--output_video",
        "test_warped.mp4",
        "--session",
        out_prefix,
    ])

    output_video = tmp_path / "output" / "test_warped.mp4"
    assert output_video.exists()
