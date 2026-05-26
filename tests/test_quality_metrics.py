"""Tests for quality_metrics module."""
import numpy as np
import pytest

from opera_align.quality_metrics import (
    interpolate_dtw_times,
    compute_ate,
    compute_ptau,
    assess_alignment_quality,
)


def test_compute_ate():
    """Test ATE computation."""
    errors = np.array([0.1, 0.2, 0.05])
    ate = compute_ate(errors)
    expected = (0.1 + 0.2 + 0.05) / 3
    assert np.isclose(ate, expected)


def test_compute_ate_empty():
    """Test ATE with empty array."""
    errors = np.array([])
    ate = compute_ate(errors)
    assert ate == 0.0


def test_compute_ptau():
    """Test Ptau computation."""
    errors = np.array([0.05, 0.15, 0.08, 0.2, 0.02])
    tau = 0.1
    ptau = compute_ptau(errors, tau)
    # errors < 0.1: [0.05, 0.08, 0.02] = 3 out of 5
    expected = 3 / 5
    assert np.isclose(ptau, expected)


def test_compute_ptau_empty():
    """Test Ptau with empty array."""
    errors = np.array([])
    ptau = compute_ptau(errors, 0.1)
    assert ptau == 0.0


def test_interpolate_dtw_times_simple():
    """Test DTW interpolation with a simple case."""
    # Reference: times [0.0, 0.1, 0.2, 0.3]
    ts_ref = np.array([0.0, 0.1, 0.2, 0.3])
    # Stream: times [0.0, 0.15, 0.25, 0.35]
    ts_stream = np.array([0.0, 0.15, 0.25, 0.35])
    # Identity DTW path: 0->0, 1->1, 2->2, 3->3
    path_ref = np.array([0, 1, 2, 3])
    path_stream = np.array([0, 1, 2, 3])
    
    control_points = np.array([0.1, 0.2])
    valid_times, pred_times = interpolate_dtw_times(
        control_points, ts_ref, ts_stream, path_ref, path_stream
    )
    
    assert len(valid_times) == 2
    assert np.allclose(valid_times, [0.1, 0.2])
    # With identity mapping, predictions should match stream times
    assert np.allclose(pred_times, [0.15, 0.25])


def test_interpolate_dtw_times_out_of_bounds():
    """Test that control points outside reference range are filtered."""
    ts_ref = np.array([1.0, 2.0, 3.0])
    ts_stream = np.array([1.0, 2.0, 3.0])
    path_ref = np.array([0, 1, 2])
    path_stream = np.array([0, 1, 2])
    
    control_points = np.array([0.5, 1.5, 3.5])  # 0.5 and 3.5 are out of bounds
    valid_times, pred_times = interpolate_dtw_times(
        control_points, ts_ref, ts_stream, path_ref, path_stream
    )
    
    assert len(valid_times) == 1
    assert np.isclose(valid_times[0], 1.5)


def test_assess_alignment_quality_basic():
    """Test full quality assessment."""
    ts_ref = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    ts_stream = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    path_ref = np.array([0, 1, 2, 3, 4])
    path_stream = np.array([0, 1, 2, 3, 4])
    
    result = assess_alignment_quality(
        ts_ref, ts_stream, path_ref, path_stream, tau=0.1
    )
    
    assert result['num_control_points'] == 5
    assert result['ate'] == 0.0
    assert result['ptau'] == 1.0
    assert result['tau'] == 0.1


def test_assess_alignment_quality_with_errors():
    """Test quality assessment with some errors."""
    ts_ref = np.array([0.0, 0.1, 0.2, 0.3])
    # Stream slightly offset
    ts_stream = np.array([0.0, 0.12, 0.21, 0.35])
    path_ref = np.array([0, 1, 2, 3])
    path_stream = np.array([0, 1, 2, 3])
    
    result = assess_alignment_quality(
        ts_ref, ts_stream, path_ref, path_stream, tau=0.1
    )
    
    assert result['num_control_points'] == 4
    assert result['ate'] > 0.0
    assert result['ptau'] < 1.0


def test_assess_alignment_quality_custom_control_points():
    """Test with custom control points."""
    ts_ref = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    ts_stream = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    path_ref = np.array([0, 1, 2, 3, 4])
    path_stream = np.array([0, 1, 2, 3, 4])
    
    control_points = np.array([0.05, 0.15, 0.25])
    result = assess_alignment_quality(
        ts_ref, ts_stream, path_ref, path_stream,
        control_point_times=control_points,
        tau=0.1
    )
    
    assert result['num_control_points'] == 3
    assert np.allclose(result['control_point_times'], control_points)
