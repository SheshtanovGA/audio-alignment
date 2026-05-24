"""End-to-end alignment, plot, and video warp pipeline."""
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import alignment, audio_features, io_utils, plotting, video_warp


@dataclass
class AlignmentResult:
    session: str
    artifact_dir: str
    ts_ref: np.ndarray
    ts_stream: np.ndarray
    path_ref: np.ndarray
    path_stream: np.ndarray
    mapped_ref_times: np.ndarray
    mapped_stream_times: np.ndarray


def run_alignment(
    ref_wav: str,
    stream_wav: str,
    session: str,
    *,
    artifacts_dir: str = "artifacts",
    sr: int = 48000,
    feature: str = "openl3",
    backend: str = "fastdtw",
    hop_size: float = 0.1,
    hop_length: Optional[int] = None,
    embedding_size: int = 512,
    n_mfcc: int = 20,
    chroma_type: str = "cqt",
) -> AlignmentResult:
    """Align stream audio to reference audio; save artifacts under artifacts_dir/session/."""
    import librosa

    openl3 = None
    if feature == "openl3":
        try:
            import openl3 as openl3_mod
        except Exception as e:
            raise RuntimeError("openl3 is required when feature='openl3'") from e
        openl3 = openl3_mod

    print("Loading audio...")
    y_ref, sr_ref = audio_features.load_audio(librosa, ref_wav, sr=sr)
    y_stream, sr_stream = audio_features.load_audio(librosa, stream_wav, sr=sr)

    if hop_length is None:
        hop_length = int(hop_size * sr)

    extract_kw = dict(
        method=feature,
        hop_size=hop_size,
        hop_length=hop_length,
        embedding_size=embedding_size,
        n_mfcc=n_mfcc,
        chroma_type=chroma_type,
        openl3=openl3,
    )

    print(f"Extracting {feature} features...")
    emb_ref, ts_ref = audio_features.extract_features(librosa, y_ref, sr_ref, **extract_kw)
    emb_stream, ts_stream = audio_features.extract_features(librosa, y_stream, sr_stream, **extract_kw)

    emb_ref = audio_features.normalize_embeddings(emb_ref)
    emb_stream = audio_features.normalize_embeddings(emb_stream)

    print(f"Computing DTW ({backend})...")
    path_ref, path_stream = alignment.compute_dtw_path(emb_ref, emb_stream, backend=backend)

    artifact_dir = os.path.join(artifacts_dir, session)
    io_utils.safe_npy_save(ts_ref, os.path.join(artifact_dir, "ts_ref.npy"))
    io_utils.safe_npy_save(ts_stream, os.path.join(artifact_dir, "ts_stream.npy"))
    io_utils.safe_npy_save(path_ref, os.path.join(artifact_dir, "path_ref.npy"))
    io_utils.safe_npy_save(path_stream, os.path.join(artifact_dir, "path_stream.npy"))

    mapped_ref_times = ts_ref[path_ref]
    mapped_stream_times = ts_stream[path_stream]

    return AlignmentResult(
        session=session,
        artifact_dir=artifact_dir,
        ts_ref=ts_ref,
        ts_stream=ts_stream,
        path_ref=path_ref,
        path_stream=path_stream,
        mapped_ref_times=mapped_ref_times,
        mapped_stream_times=mapped_stream_times,
    )


def run_pipeline(
    ref_wav: str,
    stream_wav: str,
    video: str,
    session: str,
    *,
    output_dir: str = "output",
    artifacts_dir: str = "artifacts",
    plot_name: str = "alignment.png",
    video_name: str = "warped.mp4",
    subtitles_csv: Optional[str] = None,
    warp_audio: bool = True,
    sr: int = 48000,
    feature: str = "openl3",
    backend: str = "fastdtw",
    hop_size: float = 0.1,
    hop_length: Optional[int] = None,
    embedding_size: int = 512,
    n_mfcc: int = 20,
    chroma_type: str = "cqt",
) -> dict:
    """Align audio, write alignment plot, and warp video to the reference timeline."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    result = run_alignment(
        ref_wav,
        stream_wav,
        session,
        artifacts_dir=artifacts_dir,
        sr=sr,
        feature=feature,
        backend=backend,
        hop_size=hop_size,
        hop_length=hop_length,
        embedding_size=embedding_size,
        n_mfcc=n_mfcc,
        chroma_type=chroma_type,
    )

    plot_path = os.path.join(output_dir, f"{session}_{plot_name}")
    print(f"Writing alignment plot to {plot_path}...")
    plotting.plot_alignment(
        result.ts_ref,
        result.ts_stream,
        result.path_ref,
        result.path_stream,
        subtitles_csv=subtitles_csv,
        output_png=plot_path,
    )

    video_path = os.path.join(output_dir, f"{session}_{video_name}")
    print(f"Warping video to {video_path}...")
    video_warp.warp_video(
        video,
        result.mapped_ref_times,
        result.mapped_stream_times,
        video_path,
        warp_audio=warp_audio,
    )

    return {
        "session": session,
        "artifact_dir": result.artifact_dir,
        "plot_path": plot_path,
        "video_path": video_path,
    }
