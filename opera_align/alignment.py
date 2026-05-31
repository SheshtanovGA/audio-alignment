"""DTW alignment helpers."""
from typing import Tuple
import numpy as np

def compute_dtw_path_fastdtw(fastdtw, emb_ref: np.ndarray, emb_stream: np.ndarray):
    dist, path = fastdtw(emb_ref, emb_stream, dist=cosine)
    path_ref, path_stream = zip(*path)
    return np.array(path_ref), np.array(path_stream)

def compute_dtw_path_cd(emb_ref: np.ndarray, emb_stream: np.ndarray):
    from scipy.spatial.distance import cdist
    D = cdist(emb_ref, emb_stream, metric='cosine')
    n, m = D.shape

    # cumulative cost matrix
    C = np.full((n, m), np.inf, dtype=np.float64)
    C[0, 0] = D[0, 0]

    # initialize borders
    for i in range(1, n):
        C[i, 0] = D[i, 0] + C[i - 1, 0]

    for j in range(1, m):
        C[0, j] = D[0, j] + C[0, j - 1]

    # DP
    for i in range(1, n):
        for j in range(1, m):
            C[i, j] = D[i, j] + min(
                C[i - 1, j],      # insertion
                C[i, j - 1],      # deletion
                C[i - 1, j - 1],  # match
            )

    # backtrack optimal path
    i, j = n - 1, m - 1
    path = [(i, j)]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            prev = np.argmin([
                C[i - 1, j],
                C[i, j - 1],
                C[i - 1, j - 1],
            ])

            if prev == 0:
                i -= 1
            elif prev == 1:
                j -= 1
            else:
                i -= 1
                j -= 1

        path.append((i, j))

    path.reverse()
    path_ref, path_stream = zip(*path)
    return np.array(path_ref), np.array(path_stream)

def compute_dtw_path(emb_ref: np.ndarray, emb_stream: np.ndarray, backend: str = 'fastdtw') -> Tuple[np.ndarray, np.ndarray]:
    """Compute DTW path between embedding sequences.

    backend: 'fastdtw' or 'librosa' or 'fallback'
    Returns arrays of indices (path_ref, path_stream).
    """
    if backend == 'fastdtw':
        import fastdtw
        return compute_dtw_path_fastdtw(fastdtw.fastdtw, emb_ref, emb_stream, dist=cosine)
    if backend == 'librosa':
        import librosa
        _, wp = librosa.sequence.dtw(X=emb_ref.T, Y=emb_stream.T, metric='cosine')
        path_ref, path_stream = zip(*wp[::-1])
        return np.array(path_ref), np.array(path_stream)
    return compute_dtw_path_cd(emb_ref, emb_stream)
