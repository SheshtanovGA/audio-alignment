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
    (tmp_path / "cache").mkdir(exist_ok=True)

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

    artifact_dir = tmp_path / "cache" / out_prefix
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


def test_cli_assess_quality(tmp_path, monkeypatch):
    """Test assess-quality command with two sessions."""
    pytest.importorskip("openl3")

    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "tests" / "sources"

    ref_wav = source_dir / "test_ref.wav"
    stream_wav = source_dir / "test_stream.wav"

    monkeypatch.chdir(tmp_path)

    # Create output directories
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "cache").mkdir(exist_ok=True)

    # Create two sessions with the same alignment (for testing purposes)
    session1 = "test_ref_session"
    session2 = "test_eval_session"

    # Align to create session1
    main([
        "align",
        "--ref_wav",
        str(ref_wav),
        "--stream_wav",
        str(stream_wav),
        "--out_prefix",
        session1,
    ])

    # Align again to create session2 (same input, should be similar)
    main([
        "align",
        "--ref_wav",
        str(ref_wav),
        "--stream_wav",
        str(stream_wav),
        "--out_prefix",
        session2,
    ])

    # Now assess quality of session2 against session1
    main([
        "assess-quality",
        "--ref_session",
        session1,
        "--test_session",
        session2,
        "--tau",
        "0.1",
        "--output_report",
        "quality_report.csv",
    ])

    # Check that report was generated
    report_file = tmp_path / "output" / "quality_report.csv"
    assert report_file.exists()

    # Load and verify report structure
    import pandas as pd
    df = pd.read_csv(report_file)
    assert 'control_point_index' in df.columns
    assert 'reference_time' in df.columns
    assert 'predicted_time' in df.columns
    assert 'absolute_error' in df.columns
    assert 'within_threshold' in df.columns
    assert len(df) > 0
