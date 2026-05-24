"""Simple CLI to run alignment tasks."""
import argparse
import numpy as np
import os

from . import io_utils, audio_features, alignment, subtitle_mapper, plotting, video_warp
from . import pipeline as pipeline_mod


def _alignment_paths_from_args(args) -> dict:
    return io_utils.resolve_alignment_paths(
        session=getattr(args, "session", None),
        artifacts_dir=getattr(args, "artifacts_dir", "artifacts"),
        overrides={
            "ts_ref": getattr(args, "ts_ref", None),
            "ts_stream": getattr(args, "ts_stream", None),
            "path_ref": getattr(args, "path_ref", None),
            "path_stream": getattr(args, "path_stream", None),
        },
    )


def _add_session_artifact_args(parser):
    parser.add_argument(
        "--session",
        help="Session name under artifacts/ (e.g. session3 -> artifacts/session3/ts_ref.npy)",
    )
    parser.add_argument(
        "--artifacts_dir",
        default="artifacts",
        help="Root directory for session artifacts (default: artifacts)",
    )
    for name in io_utils.ALIGNMENT_ARTIFACT_NAMES:
        parser.add_argument(
            f"--{name}",
            required=False,
            help=f"Override path to {name}.npy (default: <artifacts_dir>/<session>/{name}.npy)",
        )


def _align_kwargs_from_args(args):
    return dict(
        sr=args.sr,
        feature=args.feature,
        backend=args.backend,
        hop_size=args.hop_size,
        hop_length=args.hop_length,
        embedding_size=args.embedding_size,
        n_mfcc=args.n_mfcc,
        chroma_type=args.chroma_type,
    )


def cmd_align(args):
    artifacts_dir = getattr(args, "artifacts_dir", "artifacts")
    result = pipeline_mod.run_alignment(
        args.ref_wav,
        args.stream_wav,
        args.out_prefix,
        artifacts_dir=artifacts_dir,
        **_align_kwargs_from_args(args),
    )

    if args.subtitles:
        out_name = getattr(args, "out_subtitles", "mapped_subtitles.csv")
        out_path = os.path.join("output", out_name)
        subtitle_mapper.map_subtitles(
            args.subtitles,
            result.mapped_ref_times,
            result.mapped_stream_times,
            out_path,
        )
        print(f"Mapped subtitles written to {out_path}")


def cmd_pipeline(args):
    session = args.session or os.path.splitext(os.path.basename(args.video))[0]
    outputs = pipeline_mod.run_pipeline(
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
        **_align_kwargs_from_args(args),
    )
    print(f"Artifacts: {outputs['artifact_dir']}")
    print(f"Plot:      {outputs['plot_path']}")
    print(f"Video:     {outputs['video_path']}")


def cmd_map_subtitles(args):
    paths = _alignment_paths_from_args(args)
    ts_ref = io_utils.safe_npy_load(paths["ts_ref"])
    ts_stream = io_utils.safe_npy_load(paths["ts_stream"])
    path_ref = io_utils.safe_npy_load(paths["path_ref"])
    path_stream = io_utils.safe_npy_load(paths["path_stream"])

    mapped_ref_times = ts_ref[path_ref]
    mapped_stream_times = ts_stream[path_stream]
    subtitle_mapper.map_subtitles(args.subtitles_csv, mapped_ref_times, mapped_stream_times, "output/" + args.session + "_" + args.output_csv)
    print(f"Mapped subtitles written to output/{args.session}_{args.output_csv}")


def cmd_plot(args):
    paths = _alignment_paths_from_args(args)
    ts_ref = io_utils.safe_npy_load(paths["ts_ref"])
    ts_stream = io_utils.safe_npy_load(paths["ts_stream"])
    path_ref = io_utils.safe_npy_load(paths["path_ref"])
    path_stream = io_utils.safe_npy_load(paths["path_stream"])
    plotting.plot_alignment(ts_ref, ts_stream, path_ref, path_stream, args.subtitles_csv, "output/" + args.session + "_" + args.output_png)
    print(f"Plot written to output/{args.session}_{args.output_png}")


def cmd_warp(args):
    import pandas as pd
    if args.curve_csv:
        df = pd.read_csv(args.curve_csv)
        mapped_ref_times = df['reference_time'].values
        mapped_stream_times = df['stream_time'].values
    else:
        paths = _alignment_paths_from_args(args)
        ts_ref = io_utils.safe_npy_load(paths["ts_ref"])
        ts_stream = io_utils.safe_npy_load(paths["ts_stream"])
        path_ref = io_utils.safe_npy_load(paths["path_ref"])
        path_stream = io_utils.safe_npy_load(paths["path_stream"])
        mapped_ref_times = ts_ref[path_ref]
        mapped_stream_times = ts_stream[path_stream]

    video_warp.warp_video(
        args.input_video,
        mapped_ref_times,
        mapped_stream_times,
        "output/" + args.session + "_" + args.output_video,
        warp_audio=not args.no_audio,
    )
    print(f"Warped video written to output/{args.session}_{args.output_video}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog='opera_align')
    sub = parser.add_subparsers(dest='cmd')

    p_align = sub.add_parser('align')
    p_align.add_argument('--ref_wav', required=True)
    p_align.add_argument('--stream_wav', required=True)
    p_align.add_argument('--subtitles', required=False)
    p_align.add_argument('--out_prefix', default='artifacts/out')
    p_align.add_argument('--out_subtitles', default='artifacts/mapped_subtitles.csv')
    p_align.add_argument('--sr', type=int, default=48000)
    p_align.add_argument('--feature', choices=['openl3', 'mfcc', 'chroma'], default='openl3')
    p_align.add_argument('--embedding_size', type=int, default=512)
    p_align.add_argument('--hop_size', type=float, default=0.1, help='Frame hop in seconds (OpenL3; default hop for librosa features)')
    p_align.add_argument('--hop_length', type=int, default=None, help='Librosa hop in samples (mfcc/chroma); defaults to hop_size * sr')
    p_align.add_argument('--n_mfcc', type=int, default=20)
    p_align.add_argument('--chroma_type', choices=['cqt', 'stft'], default='cqt')
    p_align.add_argument('--backend', choices=['fastdtw', 'librosa', 'fallback'], default='fastdtw')

    p_pipeline = sub.add_parser(
        "pipeline",
        help="Align audio, plot alignment, and warp video (one command)",
    )
    p_pipeline.add_argument("--ref_wav", required=True)
    p_pipeline.add_argument("--stream_wav", required=True)
    p_pipeline.add_argument("--video", required=True, help="Performance video to warp")
    p_pipeline.add_argument("--session", default=None)
    p_pipeline.add_argument("--output_dir", default="output")
    p_pipeline.add_argument("--artifacts_dir", default="artifacts")
    p_pipeline.add_argument("--plot_name", default="alignment.png")
    p_pipeline.add_argument("--video_name", default="warped.mp4")
    p_pipeline.add_argument("--subtitles", default=None)
    p_pipeline.add_argument("--no-audio", action="store_true")
    p_pipeline.add_argument("--sr", type=int, default=48000)
    p_pipeline.add_argument("--feature", choices=["openl3", "mfcc", "chroma"], default="openl3")
    p_pipeline.add_argument("--embedding_size", type=int, default=512)
    p_pipeline.add_argument("--hop_size", type=float, default=0.1)
    p_pipeline.add_argument("--hop_length", type=int, default=None)
    p_pipeline.add_argument("--n_mfcc", type=int, default=20)
    p_pipeline.add_argument("--chroma_type", choices=["cqt", "stft"], default="cqt")
    p_pipeline.add_argument("--backend", choices=["fastdtw", "librosa", "fallback"], default="fastdtw")

    p_map = sub.add_parser('map-subtitles')
    _add_session_artifact_args(p_map)
    p_map.add_argument('--subtitles_csv', required=True)
    p_map.add_argument('--output_csv', default='mapped_subtitles.csv')

    p_plot = sub.add_parser('plot')
    _add_session_artifact_args(p_plot)
    p_plot.add_argument('--subtitles_csv', required=False)
    p_plot.add_argument('--output_png', default='alignment_plot.png')

    p_warp = sub.add_parser('warp-video')
    p_warp.add_argument('--input_video', required=True)
    p_warp.add_argument('--output_video', required=True)
    p_warp.add_argument('--curve_csv', required=False)
    p_warp.add_argument('--no-audio', action='store_true', help='Warp video only; omit audio from output')
    _add_session_artifact_args(p_warp)

    args = parser.parse_args(argv)

    if args.cmd in ('plot', 'map-subtitles', 'warp-video'):
        needs_artifacts = args.cmd != 'warp-video' or not args.curve_csv
        if needs_artifacts:
            try:
                _alignment_paths_from_args(args)
            except ValueError as e:
                parser.error(str(e))

    if args.cmd == 'align':
        cmd_align(args)
    elif args.cmd == 'pipeline':
        cmd_pipeline(args)
    elif args.cmd == 'map-subtitles':
        cmd_map_subtitles(args)
    elif args.cmd == 'plot':
        cmd_plot(args)
    elif args.cmd == 'warp-video':
        cmd_warp(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
