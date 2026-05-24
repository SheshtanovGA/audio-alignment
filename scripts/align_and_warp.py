#!/usr/bin/env python
"""Align ref/stream audio, plot the mapping, and warp a performance video.

Example:
  python scripts/align_and_warp.py \\
    --ref-wav sources/ref.wav \\
    --stream-wav sources/stream.wav \\
    --video sources/stream.mp4 \\
    --session myrun \\
    --feature mfcc \\
    --backend librosa
"""
import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from opera_align.pipeline import run_pipeline


def _default_session(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Align audio (ref vs stream), plot alignment, warp video to reference timeline.",
    )
    parser.add_argument("--ref-wav", required=True, help="Reference audio WAV")
    parser.add_argument("--stream-wav", required=True, help="Stream / performance audio WAV")
    parser.add_argument(
        "--video",
        required=True,
        help="Video to warp (performance recording; mapped to reference timeline)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Artifact folder name under artifacts/ (default: video filename stem)",
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--plot-name", default="alignment.png")
    parser.add_argument("--video-name", default="warped.mp4")
    parser.add_argument("--subtitles", default=None, help="Optional subtitles CSV for plot markers")

    parser.add_argument("--feature", choices=["openl3", "mfcc", "chroma"], default="openl3")
    parser.add_argument("--backend", choices=["fastdtw", "librosa", "fallback"], default="fastdtw")
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--hop-size", type=float, default=0.1)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--chroma-type", choices=["cqt", "stft"], default="cqt")
    parser.add_argument("--no-audio", action="store_true", help="Omit audio in warped output")

    args = parser.parse_args(argv)
    session = args.session or _default_session(args.video)

    outputs = run_pipeline(
        args.ref_wav,
        args.stream_wav,
        args.video,
        session,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        plot_name=args.plot_name,
        video_name=args.video_name,
        subtitles_csv=args.subtitles,
        warp_audio=not args.no_audio,
        sr=args.sr,
        feature=args.feature,
        backend=args.backend,
        hop_size=args.hop_size,
        hop_length=args.hop_length,
        embedding_size=args.embedding_size,
        n_mfcc=args.n_mfcc,
        chroma_type=args.chroma_type,
    )

    print("Done.")
    print(f"  artifacts: {outputs['artifact_dir']}")
    print(f"  plot:      {outputs['plot_path']}")
    print(f"  video:     {outputs['video_path']}")


if __name__ == "__main__":
    main()
