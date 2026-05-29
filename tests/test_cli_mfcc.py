from pathlib import Path

import pytest

from opera_align.cli import main


def test_cli_align_mfcc_pipeline(tmp_path, monkeypatch):
    pytest.importorskip("librosa")
    pytest.importorskip("moviepy")

    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "tests" / "sources"

    ref_wav = source_dir / "test_ref.wav"
    stream_wav = source_dir / "test_stream.wav"
    subtitles_csv = source_dir / "test_subtitles.csv"

    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "cache").mkdir(exist_ok=True)

    out_prefix = "testsession_mfcc"
    main([
        "align",
        "--feature",
        "mfcc",
        "--ref_wav",
        str(ref_wav),
        "--stream_wav",
        str(stream_wav),
        "--subtitles",
        str(subtitles_csv),
        "--out_prefix",
        out_prefix,
        "--out_subtitles",
        "mapped_subtitles_mfcc.csv",
        "--backend",
        "librosa",
    ])

    artifact_dir = tmp_path / "cache" / out_prefix
    assert (artifact_dir / "ts_ref.npy").exists()
    assert (artifact_dir / "path_ref.npy").exists()
    assert (tmp_path / "output" / "mapped_subtitles_mfcc.csv").exists()
