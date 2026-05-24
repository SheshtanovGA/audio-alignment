"""Video warping utilities using moviepy."""
import os
from typing import Optional, Tuple
import numpy as np

from .subtitle_mapper import build_time_interpolator


def prepare_warp_curve(
    mapped_ref_times: np.ndarray,
    mapped_stream_times: np.ndarray,
    *,
    max_stream_time: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a monotonic reference->stream curve safe for moviepy time warping."""
    x, y = build_time_interpolator(
        mapped_ref_times,
        mapped_stream_times,
        dedupe_stream="max",
        monotonic=True,
    )
    if len(x) == 0:
        raise ValueError("Alignment curve is empty; cannot warp video.")
    if max_stream_time is not None:
        y = np.clip(y, 0.0, max(0.0, max_stream_time))
        y = np.maximum.accumulate(y)
    return x, y


def map_ref_to_stream_times(
    t,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_stream_time: Optional[float] = None,
    ref_end: Optional[float] = None,
):
    """Map reference timeline t to stream times; accepts scalar or array (moviepy passes both)."""
    t_arr = np.asarray(t, dtype=float)
    scalar_input = (
        np.isscalar(t)
        or (isinstance(t, np.ndarray) and t_arr.ndim == 0)
    )
    flat = t_arr.reshape(-1)
    if ref_end is not None:
        flat = np.clip(flat, 0.0, float(ref_end))
    stream_t = np.interp(flat, x, y, left=float(y[0]), right=float(y[-1]))
    stream_t = np.nan_to_num(stream_t, nan=float(y[0]), posinf=float(y[-1]), neginf=0.0)
    stream_t = np.maximum(stream_t, 0.0)
    if max_stream_time is not None:
        stream_t = np.minimum(stream_t, float(max_stream_time))
    if scalar_input:
        return float(stream_t[0])
    return stream_t.reshape(t_arr.shape)


def _clip_duration_limit(duration: Optional[float], margin: float) -> Optional[float]:
    if duration is None or duration <= margin:
        return None
    return max(0.0, duration - margin)


def _attach_warped_audio(video_clip, source_audio, x, y, output_duration: float):
    """Build warped audio without moviepy's fl_time on audio (avoids buffer index bugs)."""
    try:
        from moviepy.audio.AudioClip import AudioClip
    except ImportError:
        from moviepy.editor import AudioClip

    max_audio_t = _clip_duration_limit(source_audio.duration, 2.0 / source_audio.fps)
    ref_end = float(output_duration)

    def make_frame(t):
        stream_t = map_ref_to_stream_times(
            t, x, y, max_stream_time=max_audio_t, ref_end=ref_end
        )
        stream_arr = np.atleast_1d(np.asarray(stream_t, dtype=float))
        if stream_arr.size == 1:
            return source_audio.get_frame(float(stream_arr[0]))
        out = np.zeros((stream_arr.size, source_audio.nchannels))
        for i, ts in enumerate(stream_arr):
            out[i] = source_audio.get_frame(float(ts))
        return out

    return AudioClip(make_frame, duration=ref_end, fps=source_audio.fps)


def warp_video(
    input_video: str,
    mapped_ref_times: np.ndarray,
    mapped_stream_times: np.ndarray,
    output_video: str,
    apply_to=None,
    warp_audio: bool = True,
):
    if apply_to is None:
        apply_to = ["video"]
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    try:
        from moviepy.editor import VideoFileClip
    except Exception as e:
        raise RuntimeError("moviepy is required to warp videos (and ffmpeg installed).") from e

    clip = VideoFileClip(input_video)
    max_video_t = _clip_duration_limit(clip.duration, 1e-3)
    max_audio_t = None
    if clip.audio is not None:
        max_audio_t = _clip_duration_limit(clip.audio.duration, 2.0 / clip.audio.fps)

    limits = [v for v in (max_video_t, max_audio_t) if v is not None]
    max_stream = min(limits) if limits else None

    x, y = prepare_warp_curve(mapped_ref_times, mapped_stream_times, max_stream_time=max_stream)
    ref_end = float(x[-1])

    def warp_time(t):
        return map_ref_to_stream_times(t, x, y, max_stream_time=max_video_t, ref_end=ref_end)

    warped = clip.fl_time(warp_time, apply_to=apply_to, keep_duration=False)
    warped = warped.set_duration(ref_end)

    if warp_audio and clip.audio is not None and "audio" not in apply_to:
        warped = warped.set_audio(_attach_warped_audio(warped, clip.audio, x, y, ref_end))

    warped.write_videofile(output_video, codec="libx264", audio_codec="aac")
    clip.close()
    warped.close()
    return output_video
