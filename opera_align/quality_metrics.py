"""Quality assessment metrics for alignment — ATE and Ptau computation."""
from typing import Tuple, Optional
import numpy as np


def interpolate_dtw_times(
    control_point_times: np.ndarray,
    ts_ref: np.ndarray,
    ts_stream: np.ndarray,
    path_ref: np.ndarray,
    path_stream: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate stream times at control point times using DTW path.
    
    Given control points (times) in reference domain, use the DTW alignment path
    to find corresponding times in the stream domain via linear interpolation.
    
    Args:
        control_point_times: Array of control point times in seconds (reference domain)
        ts_ref: Reference frame timestamps in seconds
        ts_stream: Stream frame timestamps in seconds
        path_ref: DTW alignment path indices in reference frames
        path_stream: DTW alignment path indices in stream frames
    
    Returns:
        (valid_control_times, interpolated_stream_times) where valid_control_times
        are those within the range of ts_ref and interpolated_stream_times are
        the corresponding interpolated stream times via DTW.
    """
    # Filter control points to those within reference time range
    min_t = ts_ref[0]
    max_t = ts_ref[-1]
    valid_mask = (control_point_times >= min_t) & (control_point_times <= max_t)
    valid_control_times = control_point_times[valid_mask]
    
    if len(valid_control_times) == 0:
        return np.array([]), np.array([])
    
    interpolated_stream_times = []
    
    for t_k in valid_control_times:
        # Find index in reference frames by interpolation
        # ts_ref is monotonically increasing
        idx_ref_float = np.interp(t_k, ts_ref, np.arange(len(ts_ref)))
        idx_ref = int(np.round(idx_ref_float))
        idx_ref = np.clip(idx_ref, 0, len(ts_ref) - 1)
        
        # Find corresponding stream index via DTW path
        # path_ref contains indices in reference, find the closest one
        distances = np.abs(path_ref - idx_ref)
        closest_path_idx = np.argmin(distances)
        idx_stream = path_stream[closest_path_idx]
        
        # Interpolate stream time at that index
        # For smoother results, linearly interpolate if idx_stream is between frames
        if idx_stream < len(ts_stream) - 1:
            # If idx_stream_float is available, use it for smoother interpolation
            t_stream = ts_stream[int(idx_stream)]
        else:
            t_stream = ts_stream[-1]
        
        interpolated_stream_times.append(t_stream)
    
    return valid_control_times, np.array(interpolated_stream_times)


def compute_ate(absolute_errors: np.ndarray) -> float:
    """Compute Average Temporal Error (ATE).
    
    ATE = (1/K) * sum(|t_k* - t_k|)
    
    Args:
        absolute_errors: Array of absolute errors in seconds
    
    Returns:
        Mean absolute error in seconds
    """
    if len(absolute_errors) == 0:
        return 0.0
    return float(np.mean(absolute_errors))


def compute_ptau(absolute_errors: np.ndarray, tau: float) -> float:
    """Compute Proportion within tau (Ptau).
    
    Ptau = (1/K) * |{k : |t_k* - t_k| < tau}|
    
    Args:
        absolute_errors: Array of absolute errors in seconds
        tau: Threshold in seconds
    
    Returns:
        Proportion of errors below threshold (0.0 to 1.0)
    """
    if len(absolute_errors) == 0:
        return 0.0
    within_threshold = np.sum(absolute_errors < tau)
    return float(within_threshold / len(absolute_errors))


def assess_alignment_quality(
    ts_ref: np.ndarray,
    ts_stream: np.ndarray,
    path_ref: np.ndarray,
    path_stream: np.ndarray,
    control_point_times: Optional[np.ndarray] = None,
    tau: float = 0.1,
) -> dict:
    """Assess alignment quality by comparing predicted vs reference times at control points.
    
    Args:
        ts_ref: Reference frame timestamps in seconds
        ts_stream: Stream frame timestamps in seconds
        path_ref: DTW alignment path indices in reference frames
        path_stream: DTW alignment path indices in stream frames
        control_point_times: Array of control point times in seconds. If None, uses ts_ref.
        tau: Threshold for Ptau in seconds (default 0.1)
    
    Returns:
        Dict with keys:
            - 'control_point_times': Array of valid control point times
            - 'predicted_times': Array of predicted stream times via DTW
            - 'absolute_errors': Array of absolute errors
            - 'ate': Average Temporal Error
            - 'ptau': Proportion within tau
            - 'tau': The threshold used
            - 'num_control_points': Number of valid control points
    """
    if control_point_times is None:
        control_point_times = ts_ref
    
    valid_times, pred_times = interpolate_dtw_times(
        control_point_times, ts_ref, ts_stream, path_ref, path_stream
    )
    
    if len(valid_times) == 0:
        return {
            'control_point_times': np.array([]),
            'predicted_times': np.array([]),
            'absolute_errors': np.array([]),
            'ate': 0.0,
            'ptau': 0.0,
            'tau': tau,
            'num_control_points': 0,
        }
    
    absolute_errors = np.abs(pred_times - valid_times)
    ate = compute_ate(absolute_errors)
    ptau = compute_ptau(absolute_errors, tau)
    
    return {
        'control_point_times': valid_times,
        'predicted_times': pred_times,
        'absolute_errors': absolute_errors,
        'ate': ate,
        'ptau': ptau,
        'tau': tau,
        'num_control_points': len(valid_times),
    }
