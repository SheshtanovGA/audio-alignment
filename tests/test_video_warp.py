import numpy as np
import pytest

from opera_align.subtitle_mapper import build_time_interpolator
from opera_align.video_warp import (
    _ref_time_at_saturation,
    map_ref_to_stream_times,
    prepare_warp_curve,
)


def test_build_time_interpolator_monotonic_max_dedupe():
    ref = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    stream = np.array([0.0, 5.0, 1.0, 0.5, 4.0])
    x, y = build_time_interpolator(ref, stream, dedupe_stream="max", monotonic=True)
    assert np.all(np.diff(x) > 0) or len(x) == 1
    assert np.all(np.diff(y) >= 0)
    assert y[1] >= 5.0
    assert y[2] >= y[1]


def test_prepare_warp_curve_clips_to_duration():
    ref = np.array([0.0, 1.0, 2.0])
    stream = np.array([0.0, 50.0, 100.0])
    x, y = prepare_warp_curve(ref, stream, max_stream_time=10.0)
    assert np.all(y <= 10.0)
    assert np.all(np.diff(y) >= 0)


def test_warp_time_never_negative_after_prepare():
    ref = np.linspace(0, 10, 50)
    stream = np.linspace(0, 10, 50)
    # local backward jumps from bad DTW
    stream[10:30] = stream[10:30] - np.linspace(0, 8, 20)
    x, y = prepare_warp_curve(ref, stream, max_stream_time=10.0)
    query = np.linspace(0, 10, 200)
    mapped = np.interp(query, x, y, left=y[0], right=y[-1])
    assert np.all(mapped >= 0.0)
    assert np.all(mapped <= 10.0)


def test_map_ref_to_stream_times_scalar_and_array():
    ref = np.array([0.0, 1.0, 2.0])
    stream = np.array([0.0, 1.0, 2.0])
    x, y = prepare_warp_curve(ref, stream, max_stream_time=10.0)
    assert map_ref_to_stream_times(0.5, x, y, max_stream_time=10.0) == pytest.approx(0.5)
    arr = np.array([0.0, 0.5, 1.0])
    out = map_ref_to_stream_times(arr, x, y, max_stream_time=10.0)
    assert out.shape == (3,)
    assert np.allclose(out, arr)


def test_map_ref_to_stream_times_clamps_negative_t():
    ref = np.array([0.0, 1.0, 2.0])
    stream = np.array([1.0, 2.0, 3.0])
    x, y = prepare_warp_curve(ref, stream, max_stream_time=10.0)
    assert map_ref_to_stream_times(-100.0, x, y, max_stream_time=10.0, ref_end=2.0) >= 0.0


def test_ref_time_at_saturation():
    ref = np.array([0.0, 10.0, 20.0, 30.0])
    stream = np.array([0.0, 5.0, 10.0, 10.0])
    x, y = prepare_warp_curve(ref, stream, max_stream_time=10.0)
    assert _ref_time_at_saturation(x, y, 10.0, eps=0.05) == pytest.approx(20.0)


def test_prepare_warp_curve_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        prepare_warp_curve(np.array([]), np.array([]))
