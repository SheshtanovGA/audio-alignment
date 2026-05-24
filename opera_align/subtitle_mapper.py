"""Map subtitle times from reference to stream using DTW mapping pairs."""
from typing import Sequence, Tuple
import numpy as np
import pandas as pd


def build_time_interpolator(
    mapped_ref_times: Sequence[float],
    mapped_stream_times: Sequence[float],
    *,
    dedupe_stream: str = "first",
    monotonic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted arrays (x, y) suitable for np.interp: reference time -> stream time.

    dedupe_stream: when reference times repeat after sorting, keep ``first`` or ``max`` stream time.
    monotonic: if True, force stream times to be non-decreasing (required for video warp).
    """
    mapped_ref_times = np.asarray(mapped_ref_times, dtype=float)
    mapped_stream_times = np.asarray(mapped_stream_times, dtype=float)
    valid = np.isfinite(mapped_ref_times) & np.isfinite(mapped_stream_times)
    mapped_ref_times = mapped_ref_times[valid]
    mapped_stream_times = mapped_stream_times[valid]
    if len(mapped_ref_times) == 0:
        return mapped_ref_times, mapped_stream_times

    order = np.argsort(mapped_ref_times)
    x = mapped_ref_times[order]
    y = mapped_stream_times[order]

    change_pts = np.where(np.diff(x) != 0)[0] + 1
    starts = np.concatenate(([0], change_pts))
    ends = np.concatenate((change_pts, [len(x)]))

    if dedupe_stream == "first":
        x = x[starts]
        y = y[starts]
    elif dedupe_stream == "max":
        x = x[starts]
        y = np.array([np.max(y[s:e]) for s, e in zip(starts, ends)])
    else:
        raise ValueError(f"Unknown dedupe_stream: {dedupe_stream!r} (use 'first' or 'max')")

    if monotonic:
        y = np.maximum.accumulate(y)
    y = np.maximum(y, 0.0)
    return x, y

def map_subtitles(subtitles_csv: str, mapped_ref_times: Sequence[float], mapped_stream_times: Sequence[float], output_csv: str):
    df = pd.read_csv(subtitles_csv)
    x, y = build_time_interpolator(mapped_ref_times, mapped_stream_times)

    starts = df['start_time'].astype(float).values
    ends = df['end_time'].astype(float).values

    mapped_starts = np.interp(starts, x, y, left=y[0], right=y[-1])
    mapped_ends = np.interp(ends, x, y, left=y[0], right=y[-1])

    df['start_time_stream'] = mapped_starts
    df['end_time_stream'] = mapped_ends
    df.to_csv(output_csv, index=False)
    return output_csv
