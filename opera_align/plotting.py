"""Plotting helpers for alignment visualization (publication style)."""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Single-column / inset-friendly defaults (IEEE-style).
_ACADEMIC_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

_FORMULATION = (

)


def _apply_academic_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)


def plot_alignment(
    ts_ref,
    ts_stream,
    path_ref,
    path_stream,
    subtitles_csv=None,
    output_png="alignment_plot.png",
):
    """Plot the optimal DTW warping path in reference vs stream time coordinates."""
    times_ref = np.asarray(ts_ref)[np.asarray(path_ref)]
    times_stream = np.asarray(ts_stream)[np.asarray(path_stream)]

    with plt.rc_context(_ACADEMIC_RC):
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        ax.plot(
            times_ref,
            times_stream,
            color="black",
            linewidth=1.2,
            label=r"$\pi^{*}$ (DTW)",
        )
        ax.set_xlabel(
            r"$t_{\mathrm{R}}(i)$, с "
        )
        ax.set_ylabel(
            r"$t_{\mathrm{S}}(\pi^{*}(i))$, с "
        )
        ax.set_title(r"$\pi^{*}$", pad=8)
        _apply_academic_axes(ax)
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="0.8")
        '''
        fig.text(
            0.5,
            0.02,
            _FORMULATION,
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
        '''
        if subtitles_csv:
            df_sub = pd.read_csv(subtitles_csv)
            for _, row in df_sub.iterrows():
                ax.axvline(
                    x=row["start_time"],
                    color="0.35",
                    linestyle=":",
                    linewidth=0.7,
                    alpha=0.6,
                )
                ax.axvline(
                    x=row["end_time"],
                    color="0.55",
                    linestyle=":",
                    linewidth=0.7,
                    alpha=0.6,
                )
            handles, labels = ax.get_legend_handles_labels()
            handles.extend(
                [
                    plt.Line2D([0], [0], color="0.35", linestyle=":", linewidth=0.7),
                    plt.Line2D([0], [0], color="0.55", linestyle=":", linewidth=0.7),
                ]
            )
            ax.legend(
                handles,
                labels,
                loc="upper left",
                frameon=True,
                framealpha=0.9,
                edgecolor="0.8",
            )

        fig.subplots_adjust(bottom=0.22)
        fig.savefig(output_png)
        plt.close(fig)

    return output_png, times_ref, times_stream
