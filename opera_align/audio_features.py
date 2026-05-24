"""Audio loading and embedding extraction helpers."""
from typing import Tuple
import numpy as np

def load_audio(librosa, file_path: str, sr: int = 48000) -> Tuple[np.ndarray, int]:
    """Load audio file and return mono signal and sample rate.

    `librosa` is injected to make testing and optional imports easier.
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr

def extract_openl3_embeddings(openl3, audio: np.ndarray, sr: int, embedding_size: int = 512, hop_size: float = 0.1):
    """Extract OpenL3 embeddings. Raises ImportError if OpenL3 missing."""
    emb, ts = openl3.get_audio_embedding(audio, sr, embedding_size=embedding_size,
                                         hop_size=hop_size, center=True, verbose=False)
    return emb, ts

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    return embeddings / norm
