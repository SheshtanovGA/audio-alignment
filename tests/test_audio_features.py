import numpy as np
import pytest

from opera_align import alignment, audio_features


@pytest.fixture
def librosa_mod():
    pytest.importorskip("librosa")
    import librosa
    return librosa


def test_mfcc_extractor_shape(librosa_mod):
    sr = 48000
    hop_length = 4800
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    features, ts = audio_features.extract_mfcc_features(
        librosa_mod, audio, sr, hop_length=hop_length, n_mfcc=20
    )

    assert features.shape[1] == 20
    assert features.shape[0] == len(ts)
    assert np.all(np.diff(ts) > 0)


def test_chroma_extractor_shape(librosa_mod):
    sr = 48000
    hop_length = 4800
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    features, ts = audio_features.extract_chroma_features(
        librosa_mod, audio, sr, hop_length=hop_length, chroma_type="cqt"
    )

    assert features.shape[1] == 12
    assert features.shape[0] == len(ts)
    assert np.all(np.diff(ts) > 0)


def test_extract_features_dispatcher_mfcc(librosa_mod):
    sr = 48000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 330 * t)

    features, ts = audio_features.extract_features(
        librosa_mod, audio, sr, method="mfcc", hop_size=0.1, n_mfcc=13
    )

    assert features.shape[1] == 13
    assert len(ts) == features.shape[0]


def test_dtw_detects_constant_frame_offset(librosa_mod):
    sr = 48000
    hop_length = 4800
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    base = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)

    offset_samples = 3 * hop_length
    delayed = np.concatenate([np.zeros(offset_samples), base])[: len(base)]

    feat_ref, _ = audio_features.extract_mfcc_features(
        librosa_mod, base, sr, hop_length=hop_length, n_mfcc=20
    )
    feat_stream, _ = audio_features.extract_mfcc_features(
        librosa_mod, delayed, sr, hop_length=hop_length, n_mfcc=20
    )
    feat_ref = audio_features.normalize_embeddings(feat_ref)
    feat_stream = audio_features.normalize_embeddings(feat_stream)

    path_ref, path_stream = alignment.compute_dtw_path(
        feat_ref, feat_stream, backend="librosa"
    )

    median_offset = int(np.median(path_stream - path_ref))
    assert abs(median_offset - 3) <= 2
