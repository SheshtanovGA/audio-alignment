"""Video warping utilities using moviepy."""
import os
import warnings
from typing import Any, Dict, Optional, Tuple
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


def _probe_safe_end_time(clip, candidate_max: float, min_time: float = 0.0) -> float:
    """Binary-search the latest stream time where get_frame still succeeds."""
    candidate_max = float(candidate_max)
    min_time = float(min_time)
    if candidate_max <= min_time:
        return min_time

    try:
        clip.get_frame(min_time)
    except Exception:
        return 0.0

    lo, hi = min_time, candidate_max
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        try:
            clip.get_frame(mid)
            lo = mid
        except Exception:
            hi = mid
        if hi - lo < 0.05:
            break
    return lo


def _ref_time_at_saturation(x: np.ndarray, y: np.ndarray, max_stream_time: float, eps: float = 0.05) -> float:
    """Reference time where stream mapping first reaches the stream ceiling."""
    ceiling = max(0.0, float(max_stream_time) - eps)
    hits = np.where(y >= ceiling)[0]
    if len(hits) == 0:
        return float(x[-1])
    return float(x[hits[0]])


def _make_safe_video_fl(
    x: np.ndarray,
    y: np.ndarray,
    max_safe_t: float,
    ref_end: float,
) -> Any:
    """Return a moviepy fl() callback that never seeks past a verified readable time."""
    state: Dict[str, Any] = {"frame": None}

    def fl(gf, t):
        stream_t = map_ref_to_stream_times(
            t, x, y, max_stream_time=max_safe_t, ref_end=ref_end
        )
        st_arr = np.atleast_1d(np.asarray(stream_t, dtype=float))
        if st_arr.size == 1:
            st_val = float(st_arr[0])
            try:
                frame = gf(st_val)
                state["frame"] = frame
                return frame
            except Exception:
                if state["frame"] is not None:
                    return state["frame"]
                try:
                    frame = gf(0.0)
                    state["frame"] = frame
                    return frame
                except Exception as e:
                    raise OSError(
                        f"Could not read video frame at t={st_val:.3f}s "
                        f"(safe limit {max_safe_t:.3f}s)"
                    ) from e

        out = []
        for st_val in st_arr:
            try:
                frame = gf(float(st_val))
                state["frame"] = frame
            except Exception:
                frame = state["frame"]
                if frame is None:
                    frame = gf(0.0)
                    state["frame"] = frame
            out.append(frame)
        return np.array(out)

    return fl


def _attach_warped_audio(source_audio, x, y, output_duration: float, max_audio_t: float):
    """Build warped audio without moviepy's fl_time on audio (avoids buffer index bugs)."""
    try:
        from moviepy.audio.AudioClip import AudioClip
    except ImportError:
        from moviepy.editor import AudioClip

    ref_end = float(output_duration)
    hold_last = None
    for t_probe in (max_audio_t, max_audio_t * 0.95, 0.0):
        try:
            hold_last = source_audio.get_frame(float(t_probe))
            break
        except Exception:
            continue

    def make_frame(t):
        stream_t = map_ref_to_stream_times(
            t, x, y, max_stream_time=max_audio_t, ref_end=ref_end
        )
        stream_arr = np.atleast_1d(np.asarray(stream_t, dtype=float))
        if stream_arr.size == 1:
            ts = float(stream_arr[0])
            try:
                return source_audio.get_frame(ts)
            except Exception:
                if hold_last is not None:
                    return hold_last
                return source_audio.get_frame(0.0)
        out = np.zeros((stream_arr.size, source_audio.nchannels))
        for i, ts in enumerate(stream_arr):
            try:
                out[i] = source_audio.get_frame(float(ts))
            except Exception:
                if hold_last is not None:
                    out[i] = hold_last
                else:
                    out[i] = source_audio.get_frame(0.0)
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
    nominal_video_t = _clip_duration_limit(clip.duration, 0.25)
    max_audio_t = None
    if clip.audio is not None:
        max_audio_t = _clip_duration_limit(clip.audio.duration, 2.0 / clip.audio.fps)

    if nominal_video_t is not None:
        max_video_t = _probe_safe_end_time(clip, nominal_video_t)
    else:
        max_video_t = 0.0

    if max_video_t <= 0:
        clip.close()
        raise ValueError(f"Could not read any frames from video: {input_video}")

    limits = [max_video_t]
    if max_audio_t is not None:
        limits.append(max_audio_t)
    max_stream = min(limits)

    x, y = prepare_warp_curve(mapped_ref_times, mapped_stream_times, max_stream_time=max_stream)
    ref_end = float(x[-1])

    sat_ref = _ref_time_at_saturation(x, y, max_stream)
    if ref_end > 0 and sat_ref < ref_end - 1.0:
        plateau_pct = 100.0 * (ref_end - sat_ref) / ref_end
        warnings.warn(
            f"Alignment hits stream ceiling at reference t={sat_ref:.1f}s "
            f"({plateau_pct:.1f}% of output will hold the last readable frame). "
            "Reference audio is longer than the stream video coverage.",
            UserWarning,
        )

    warped = clip.fl(
        _make_safe_video_fl(x, y, max_video_t, ref_end),
        apply_to=apply_to,
        keep_duration=False,
    )
    warped = warped.set_duration(ref_end)

    if warp_audio and clip.audio is not None and "audio" not in apply_to:
        safe_audio_t = _probe_safe_end_time(clip.audio, max_audio_t or max_video_t)
        warped = warped.set_audio(
            _attach_warped_audio(clip.audio, x, y, ref_end, safe_audio_t)
        )

    warped.write_videofile(output_video, codec="libx264", audio_codec="aac")
    clip.close()
    warped.close()
    return output_video
