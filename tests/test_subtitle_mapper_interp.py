import numpy as np

from opera_align.subtitle_mapper import build_time_interpolator


def test_build_time_interpolator_first_default():
    ref = np.array([0.0, 1.0, 1.0, 2.0])
    stream = np.array([0.0, 5.0, 1.0, 2.0])
    x, y = build_time_interpolator(ref, stream)
    assert len(x) == 3
    assert y[1] == 5.0
