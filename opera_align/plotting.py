"""Plotting helpers for alignment visualization."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_alignment(ts_ref, ts_stream, path_ref, path_stream, subtitles_csv=None, output_png="alignment_plot.png"):
    times_ref = np.asarray(ts_ref)[np.asarray(path_ref)]
    times_stream = np.asarray(ts_stream)[np.asarray(path_stream)]

    plt.figure(figsize=(12, 6))
    plt.plot(times_ref, times_stream, color='blue', linewidth=2, label="DTW Alignment")
    plt.xlabel("Reference Time (s)")
    plt.ylabel("Stream Time (s)")
    plt.title("Reference vs Live Performance Alignment Timeline")
    plt.grid(True)
    plt.legend()

    if subtitles_csv:
        df_sub = pd.read_csv(subtitles_csv)
        for idx, row in df_sub.iterrows():
            plt.axvline(x=row['start_time'], color='green', linestyle='--', alpha=0.5)
            plt.axvline(x=row['end_time'], color='red', linestyle='--', alpha=0.5)
        plt.text(0.01, 0.95, "Green: subtitle start\nRed: subtitle end", transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    return output_png, times_ref, times_stream
