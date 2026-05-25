"""Web UI builds the same argument namespace defaults as the CLI."""
import argparse
from unittest.mock import MagicMock, patch

import pytest

from opera_align import defaults as D
from opera_align.web import _build_namespace, _validate_namespace


def _mock_request(form, files=None):
    req = MagicMock()
    req.form = form
    req.files = files or {}
    return req


@pytest.mark.parametrize("cmd", ["align", "pipeline", "map-subtitles", "plot", "warp-video"])
def test_build_namespace_uses_cli_defaults(cmd):
    form = {
        "out_prefix": D.ALIGN_OUT_PREFIX,
        "out_subtitles": D.ALIGN_OUT_SUBTITLES,
        "output_dir": D.PIPELINE_OUTPUT_DIR,
        "artifacts_dir": D.ARTIFACTS_DIR,
        "plot_name": D.PIPELINE_PLOT_NAME,
        "video_name": D.PIPELINE_VIDEO_NAME,
        "output_csv": D.MAP_OUTPUT_CSV,
        "output_png": D.PLOT_OUTPUT_PNG,
        "sr": str(D.SR),
        "feature": D.FEATURE,
        "embedding_size": str(D.EMBEDDING_SIZE),
        "hop_size": str(D.HOP_SIZE),
        "n_mfcc": str(D.N_MFCC),
        "chroma_type": D.CHROMA_TYPE,
        "backend": D.BACKEND,
        "session": "testsession",
        "output_video": "out.mp4",
    }
    uploads = {
        "ref_wav": "/tmp/ref.wav",
        "stream_wav": "/tmp/stream.wav",
        "video": "/tmp/v.mp4",
        "subtitles_csv": "/tmp/subs.csv",
        "input_video": "/tmp/in.mp4",
    }
    with patch("opera_align.web.request", _mock_request(form)):
        ns = _build_namespace(cmd, uploads)
    assert isinstance(ns, argparse.Namespace)
    if cmd in ("align", "pipeline"):
        assert ns.sr == D.SR
        assert ns.feature == D.FEATURE
        assert ns.backend == D.BACKEND
        assert ns.hop_length is None
    if cmd == "align":
        assert ns.out_prefix == D.ALIGN_OUT_PREFIX
    if cmd == "pipeline":
        assert ns.output_dir == D.PIPELINE_OUTPUT_DIR
        assert ns.no_audio is False
    if cmd == "map-subtitles":
        assert ns.output_csv == D.MAP_OUTPUT_CSV
    if cmd == "plot":
        assert ns.output_png == D.PLOT_OUTPUT_PNG


def test_validate_align_requires_wav():
    ns = argparse.Namespace(ref_wav="", stream_wav="")
    assert _validate_namespace("align", ns) is not None
