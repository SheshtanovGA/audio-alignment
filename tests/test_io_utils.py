import os

import pytest

from opera_align import io_utils


def test_session_artifact_paths():
    paths = io_utils.session_artifact_paths("session3", cache_dir="cache")
    assert paths["ts_ref"] == os.path.join("cache", "session3", "ts_ref.npy")
    assert paths["path_stream"] == os.path.join("cache", "session3", "path_stream.npy")


def test_resolve_alignment_paths_from_session():
    paths = io_utils.resolve_alignment_paths(session="session2")
    assert paths["ts_stream"] == os.path.join("cache", "session2", "ts_stream.npy")


def test_resolve_alignment_paths_override():
    paths = io_utils.resolve_alignment_paths(
        session="session2",
        overrides={"path_ref": "custom/path_ref.npy"},
    )
    assert paths["path_ref"] == "custom/path_ref.npy"
    assert paths["ts_ref"] == os.path.join("cache", "session2", "ts_ref.npy")


def test_resolve_alignment_paths_explicit_only():
    paths = io_utils.resolve_alignment_paths(
        overrides={
            "ts_ref": "a/ts_ref.npy",
            "ts_stream": "a/ts_stream.npy",
            "path_ref": "a/path_ref.npy",
            "path_stream": "a/path_stream.npy",
        },
    )
    assert paths["ts_ref"] == "a/ts_ref.npy"


def test_resolve_alignment_paths_missing():
    with pytest.raises(ValueError, match="Missing alignment cache"):
        io_utils.resolve_alignment_paths(session="s", overrides={"ts_ref": "only.npy"})
