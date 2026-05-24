"""Video warping utilities using moviepy."""
import os
import numpy as np

def warp_video(input_video: str, mapped_ref_times: np.ndarray, mapped_stream_times: np.ndarray, output_video: str, apply_to=None):
    if apply_to is None:
        apply_to = ['video', 'audio']
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    try:
        from moviepy.editor import VideoFileClip
    except Exception as e:
        raise RuntimeError("moviepy is required to warp videos (and ffmpeg installed).") from e

    clip = VideoFileClip(input_video)

    # Ensure sorted reference times
    order = np.argsort(mapped_ref_times)
    x = np.asarray(mapped_ref_times)[order]
    y = np.asarray(mapped_stream_times)[order]

    def warp_time(t):
        return np.interp(t, x, y, left=y[0], right=y[-1])

    warped = clip.fl_time(warp_time, apply_to=apply_to)
    # set duration to reference's last timestamp if available
    try:
        warped = warped.set_duration(float(x[-1]))
    except Exception:
        pass

    warped.write_videofile(output_video, codec='libx264', audio_codec='aac')
    return output_video
