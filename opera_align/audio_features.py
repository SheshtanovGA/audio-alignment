"""Audio loading and feature extraction helpers."""
from typing import Optional, Tuple
import numpy as np


def load_audio(librosa, file_path: str, sr: int = 48000) -> Tuple[np.ndarray, int]:
    """Load audio file and return mono signal and sample rate.

    `librosa` is injected to make testing and optional imports easier.
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr


def frame_timestamps(librosa, n_frames: int, sr: int, hop_length: int) -> np.ndarray:
    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)


def extract_mfcc_features(
    librosa,
    audio: np.ndarray,
    sr: int,
    *,
    hop_length: int = 4800,
    n_fft: int = 2048,
    n_mels: int = 128,
    n_mfcc: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return MFCC frame matrix (T, n_mfcc) and per-frame timestamps."""
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_S = librosa.power_to_db(S)
    mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=n_mfcc)
    features = mfcc.T
    ts = frame_timestamps(librosa, features.shape[0], sr, hop_length)
    return features, ts


def extract_chroma_features(
    librosa,
    audio: np.ndarray,
    sr: int,
    *,
    hop_length: int = 4800,
    n_chroma: int = 12,
    chroma_type: str = "cqt",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return chroma frame matrix (T, n_chroma) and per-frame timestamps."""
    if chroma_type == "cqt":
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=sr, hop_length=hop_length, n_chroma=n_chroma
        )
    elif chroma_type == "stft":
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, hop_length=hop_length, n_chroma=n_chroma
        )
    else:
        raise ValueError(f"Unknown chroma_type: {chroma_type!r} (use 'cqt' or 'stft')")
    features = chroma.T
    ts = frame_timestamps(librosa, features.shape[0], sr, hop_length)
    return features, ts


def extract_openl3_embeddings(
    openl3,
    audio: np.ndarray,
    sr: int,
    embedding_size: int = 512,
    hop_size: float = 0.1,
):
    """Extract OpenL3 embeddings. Raises ImportError if OpenL3 missing."""
    emb, ts = openl3.get_audio_embedding(
        audio,
        sr,
        embedding_size=embedding_size,
        hop_size=hop_size,
        center=True,
        verbose=False,
    )
    return emb, ts


def extract_features(
    librosa,
    audio: np.ndarray,
    sr: int,
    *,
    method: str = "openl3",
    hop_length: Optional[int] = None,
    hop_size: float = 0.1,
    openl3=None,
    embedding_size: int = 512,
    n_mfcc: int = 20,
    chroma_type: str = "cqt",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract frame features for alignment.

    Returns (features[T, D], timestamps[T]).
    """
    if method == "openl3":
        if openl3 is None:
            import openl3 as openl3_mod
            openl3 = openl3_mod
        return extract_openl3_embeddings(
            openl3, audio, sr, embedding_size=embedding_size, hop_size=hop_size
        )
    if hop_length is None:
        hop_length = int(hop_size * sr)
    if method == "mfcc":
        return extract_mfcc_features(
            librosa, audio, sr, hop_length=hop_length, n_mfcc=n_mfcc
        )
    if method == "chroma":
        return extract_chroma_features(
            librosa, audio, sr, hop_length=hop_length, chroma_type=chroma_type
        )
    raise ValueError(f"Unknown feature method: {method!r} (use 'openl3', 'mfcc', or 'chroma')")


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    return embeddings / norm
