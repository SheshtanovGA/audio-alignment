"""DTW alignment helpers."""
from typing import Tuple
import numpy as np

def compute_dtw_path_fastdtw(fastdtw, emb_ref: np.ndarray, emb_stream: np.ndarray):
    dist, path = fastdtw(emb_ref, emb_stream, dist=lambda x, y: np.linalg.norm(x - y))
    path_ref, path_stream = zip(*path)
    return np.array(path_ref), np.array(path_stream)

def compute_dtw_path_cd(emb_ref: np.ndarray, emb_stream: np.ndarray):
    # Fallback naive implementation using scipy cdist + simple greedy path (not optimal)
    from scipy.spatial.distance import cdist
    D = cdist(emb_ref, emb_stream, metric='cosine')
    # Use a simple global dtw via librosa if available
    try:
        import librosa
        _, wp = librosa.sequence.dtw(X=emb_ref.T, Y=emb_stream.T, metric='cosine')
        path_ref, path_stream = zip(*wp[::-1])
        return np.array(path_ref), np.array(path_stream)
    except Exception:
        # fallback: pick minimal along diagonal-like greedy
        i = j = 0
        path = []
        n, m = D.shape
        while i < n and j < m:
            path.append((i, j))
            if i == n - 1:
                j += 1
            elif j == m - 1:
                i += 1
            else:
                # choose minimal neighbor
                choices = [(D[i + 1, j], (i + 1, j)), (D[i, j + 1], (i, j + 1)), (D[i + 1, j + 1], (i + 1, j + 1))]
                choice = min(choices, key=lambda x: x[0])[1]
                i, j = choice
        path_ref, path_stream = zip(*path)
        return np.array(path_ref), np.array(path_stream)

def compute_dtw_path(emb_ref: np.ndarray, emb_stream: np.ndarray, backend: str = 'fastdtw') -> Tuple[np.ndarray, np.ndarray]:
    """Compute DTW path between embedding sequences.

    backend: 'fastdtw' or 'librosa' or 'fallback'
    Returns arrays of indices (path_ref, path_stream).
    """
    if backend == 'fastdtw':
        try:
            import fastdtw
            return compute_dtw_path_fastdtw(fastdtw.fastdtw, emb_ref, emb_stream)
        except Exception:
            # fall through
            pass
    if backend == 'librosa':
        try:
            import librosa
            _, wp = librosa.sequence.dtw(X=emb_ref.T, Y=emb_stream.T, metric='cosine')
            path_ref, path_stream = zip(*wp[::-1])
            return np.array(path_ref), np.array(path_stream)
        except Exception:
            pass
    # final fallback
    return compute_dtw_path_cd(emb_ref, emb_stream)
