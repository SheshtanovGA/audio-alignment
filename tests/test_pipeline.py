from pathlib import Path
from unittest.mock import patch

import pytest

from opera_align.pipeline import run_pipeline


def test_run_pipeline_calls_steps(tmp_path, monkeypatch):
    pytest.importorskip("librosa")

    repo_root = Path(__file__).resolve().parents[1]
    source_dir = repo_root / "tests" / "sources"
    ref_wav = source_dir / "test_ref.wav"
    stream_wav = source_dir / "test_stream.wav"
    video = source_dir / "test_video.mp4"

    if not all(p.exists() for p in (ref_wav, stream_wav, video)):
        pytest.skip("test sources missing")

    monkeypatch.chdir(tmp_path)
    plot_calls = []
    warp_calls = []

    def fake_plot(*args, **kwargs):
        plot_calls.append(kwargs.get("output_png"))
        return kwargs.get("output_png")

    def fake_warp(input_video, mapped_ref, mapped_stream, output_video, **kwargs):
        warp_calls.append(output_video)
        Path(output_video).parent.mkdir(parents=True, exist_ok=True)
        Path(output_video).write_bytes(b"")
        return output_video

    with patch("opera_align.pipeline.plotting.plot_alignment", side_effect=fake_plot):
        with patch("opera_align.pipeline.video_warp.warp_video", side_effect=fake_warp):
            outputs = run_pipeline(
                str(ref_wav),
                str(stream_wav),
                str(video),
                "testsession",
                output_dir=str(tmp_path / "output"),
                artifacts_dir=str(tmp_path / "artifacts"),
                feature="mfcc",
                backend="librosa",
                warp_audio=False,
            )

    assert (tmp_path / "artifacts" / "testsession" / "ts_ref.npy").exists()
    assert plot_calls
    assert warp_calls
    assert Path(outputs["plot_path"]).name == "testsession_alignment.png"
    assert Path(outputs["video_path"]).name == "testsession_warped.mp4"
