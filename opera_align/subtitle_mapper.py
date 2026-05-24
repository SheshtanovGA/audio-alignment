"""Map subtitle times from reference to stream using DTW mapping pairs."""
from typing import Sequence
import numpy as np
import pandas as pd

def build_time_interpolator(mapped_ref_times: Sequence[float], mapped_stream_times: Sequence[float]):
    """Return sorted arrays (x,y) suitable for np.interp: x -> y mapping.

    This sorts by reference times and collapses duplicates by taking the first occurrence.
    """
    mapped_ref_times = np.asarray(mapped_ref_times)
    mapped_stream_times = np.asarray(mapped_stream_times)
    order = np.argsort(mapped_ref_times)
    x = mapped_ref_times[order]
    y = mapped_stream_times[order]
    # Remove duplicates in x by keeping the first
    unique_idx = np.concatenate(([0], np.where(np.diff(x) != 0)[0] + 1))
    return x[unique_idx], y[unique_idx]

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
