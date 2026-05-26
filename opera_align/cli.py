"""Simple CLI to run alignment tasks."""
import argparse
import numpy as np
import os
import pandas as pd

from . import io_utils, audio_features, alignment, subtitle_mapper, plotting, video_warp, quality_metrics
from . import pipeline as pipeline_mod
from . import defaults as D


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
        default=D.ARTIFACTS_DIR,
        help=f"Root directory for session artifacts (default: {D.ARTIFACTS_DIR})",
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


def cmd_assess_quality(args):
    """Assess alignment quality by comparing reference and test sessions."""
    # Load reference session artifacts
    ref_paths = io_utils.resolve_alignment_paths(
        session=args.ref_session,
        artifacts_dir=args.artifacts_dir,
        overrides={
            "ts_ref": getattr(args, "ref_ts_ref", None),
            "ts_stream": getattr(args, "ref_ts_stream", None),
            "path_ref": getattr(args, "ref_path_ref", None),
            "path_stream": getattr(args, "ref_path_stream", None),
        },
    )
    
    # Load test session artifacts
    test_paths = io_utils.resolve_alignment_paths(
        session=args.test_session,
        artifacts_dir=args.artifacts_dir,
        overrides={
            "ts_ref": getattr(args, "test_ts_ref", None),
            "ts_stream": getattr(args, "test_ts_stream", None),
            "path_ref": getattr(args, "test_path_ref", None),
            "path_stream": getattr(args, "test_path_stream", None),
        },
    )
    
    # Load all necessary arrays
    ts_ref_ref = io_utils.safe_npy_load(ref_paths["ts_ref"])
    ts_stream_ref = io_utils.safe_npy_load(ref_paths["ts_stream"])
    path_ref_ref = io_utils.safe_npy_load(ref_paths["path_ref"])
    path_stream_ref = io_utils.safe_npy_load(ref_paths["path_stream"])
    
    ts_ref_test = io_utils.safe_npy_load(test_paths["ts_ref"])
    ts_stream_test = io_utils.safe_npy_load(test_paths["ts_stream"])
    path_ref_test = io_utils.safe_npy_load(test_paths["path_ref"])
    path_stream_test = io_utils.safe_npy_load(test_paths["path_stream"])
    
    # Load control points (default: reference session timestamps)
    control_point_times = None
    if args.control_points:
        if args.control_points.endswith('.csv'):
            df = io_utils.safe_csv_read(args.control_points)
            control_point_times = df['time'].values if 'time' in df.columns else df.iloc[:, 0].values
        else:
            # Assume it's a path to a .npy file
            control_point_times = io_utils.safe_npy_load(args.control_points)
    
    # Assess quality of test session using reference as ground truth
    # The reference session tells us the "true" stream times
    # For each control point time in reference, we check if the test session predicts it correctly
    result = quality_metrics.assess_alignment_quality(
        ts_ref=ts_ref_ref,
        ts_stream=ts_stream_ref,
        path_ref=path_ref_ref,
        path_stream=path_stream_ref,
        control_point_times=control_point_times,
        tau=args.tau,
    )
    
    # Generate output report CSV
    report_data = {
        'control_point_index': np.arange(result['num_control_points']),
        'reference_time': result['control_point_times'],
        'predicted_time': result['predicted_times'],
        'absolute_error': result['absolute_errors'],
        'within_threshold': result['absolute_errors'] < result['tau'],
    }
    df_report = pd.DataFrame(report_data)
    
    # Create output directory if needed
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    
    # Write CSV report
    report_path = os.path.join(out_dir, args.output_report)
    df_report.to_csv(report_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Alignment Quality Assessment")
    print(f"{'='*60}")
    print(f"Reference Session: {args.ref_session}")
    print(f"Test Session: {args.test_session}")
    print(f"Number of Control Points: {result['num_control_points']}")
    print(f"Threshold (tau): {result['tau']:.4f} seconds")
    print(f"\nResults:")
    print(f"  Average Temporal Error (ATE): {result['ate']:.6f} seconds")
    print(f"  Proportion within tau (Ptau): {result['ptau']:.4f} ({result['ptau']*100:.2f}%)")
    print(f"\nDetailed report written to: {report_path}")
    print(f"{'='*60}\n")



def main(argv=None):
    parser = argparse.ArgumentParser(prog='opera_align')
    sub = parser.add_subparsers(dest='cmd')

    p_align = sub.add_parser('align')
    p_align.add_argument('--ref_wav', required=True)
    p_align.add_argument('--stream_wav', required=True)
    p_align.add_argument('--subtitles', required=False)
    p_align.add_argument('--out_prefix', default=D.ALIGN_OUT_PREFIX)
    p_align.add_argument('--out_subtitles', default=D.ALIGN_OUT_SUBTITLES)
    p_align.add_argument('--sr', type=int, default=D.SR)
    p_align.add_argument('--feature', choices=list(D.FEATURE_CHOICES), default=D.FEATURE)
    p_align.add_argument('--embedding_size', type=int, default=D.EMBEDDING_SIZE)
    p_align.add_argument('--hop_size', type=float, default=D.HOP_SIZE, help='Frame hop in seconds (OpenL3; default hop for librosa features)')
    p_align.add_argument('--hop_length', type=int, default=D.HOP_LENGTH, help='Librosa hop in samples (mfcc/chroma); defaults to hop_size * sr')
    p_align.add_argument('--n_mfcc', type=int, default=D.N_MFCC)
    p_align.add_argument('--chroma_type', choices=list(D.CHROMA_TYPE_CHOICES), default=D.CHROMA_TYPE)
    p_align.add_argument('--backend', choices=list(D.BACKEND_CHOICES), default=D.BACKEND)

    p_pipeline = sub.add_parser(
        "pipeline",
        help="Align audio, plot alignment, and warp video (one command)",
    )
    p_pipeline.add_argument("--ref_wav", required=True)
    p_pipeline.add_argument("--stream_wav", required=True)
    p_pipeline.add_argument("--video", required=True, help="Performance video to warp")
    p_pipeline.add_argument("--session", default=None)
    p_pipeline.add_argument("--output_dir", default=D.PIPELINE_OUTPUT_DIR)
    p_pipeline.add_argument("--artifacts_dir", default=D.PIPELINE_ARTIFACTS_DIR)
    p_pipeline.add_argument("--plot_name", default=D.PIPELINE_PLOT_NAME)
    p_pipeline.add_argument("--video_name", default=D.PIPELINE_VIDEO_NAME)
    p_pipeline.add_argument("--subtitles", default=None)
    p_pipeline.add_argument("--no-audio", action="store_true")
    p_pipeline.add_argument("--sr", type=int, default=D.SR)
    p_pipeline.add_argument("--feature", choices=list(D.FEATURE_CHOICES), default=D.FEATURE)
    p_pipeline.add_argument("--embedding_size", type=int, default=D.EMBEDDING_SIZE)
    p_pipeline.add_argument("--hop_size", type=float, default=D.HOP_SIZE)
    p_pipeline.add_argument("--hop_length", type=int, default=D.HOP_LENGTH)
    p_pipeline.add_argument("--n_mfcc", type=int, default=D.N_MFCC)
    p_pipeline.add_argument("--chroma_type", choices=list(D.CHROMA_TYPE_CHOICES), default=D.CHROMA_TYPE)
    p_pipeline.add_argument("--backend", choices=list(D.BACKEND_CHOICES), default=D.BACKEND)

    p_map = sub.add_parser('map-subtitles')
    _add_session_artifact_args(p_map)
    p_map.add_argument('--subtitles_csv', required=True)
    p_map.add_argument('--output_csv', default=D.MAP_OUTPUT_CSV)

    p_plot = sub.add_parser('plot')
    _add_session_artifact_args(p_plot)
    p_plot.add_argument('--subtitles_csv', required=False)
    p_plot.add_argument('--output_png', default=D.PLOT_OUTPUT_PNG)

    p_warp = sub.add_parser('warp-video')
    p_warp.add_argument('--input_video', required=True)
    p_warp.add_argument('--output_video', required=True)
    p_warp.add_argument('--curve_csv', required=False)
    p_warp.add_argument('--no-audio', action='store_true', help='Warp video only; omit audio from output')
    _add_session_artifact_args(p_warp)

    p_assess = sub.add_parser(
        'assess-quality',
        help='Assess alignment quality by comparing reference and test sessions'
    )
    p_assess.add_argument(
        '--ref_session',
        required=True,
        help='Reference session name (ground truth alignment)',
    )
    p_assess.add_argument(
        '--test_session',
        required=True,
        help='Test session name (alignment to evaluate)',
    )
    p_assess.add_argument(
        '--artifacts_dir',
        default=D.ARTIFACTS_DIR,
        help=f'Root directory for session artifacts (default: {D.ARTIFACTS_DIR})',
    )
    # Allow explicit path overrides for reference session
    for name in io_utils.ALIGNMENT_ARTIFACT_NAMES:
        p_assess.add_argument(
            f"--ref_{name}",
            required=False,
            help=f"Override path to reference {name}.npy",
        )
    # Allow explicit path overrides for test session
    for name in io_utils.ALIGNMENT_ARTIFACT_NAMES:
        p_assess.add_argument(
            f"--test_{name}",
            required=False,
            help=f"Override path to test {name}.npy",
        )
    p_assess.add_argument(
        '--control_points',
        required=False,
        help='Path to control points CSV or NPY file (default: use reference session timestamps)',
    )
    p_assess.add_argument(
        '--tau',
        type=float,
        default=D.QUALITY_ASSESS_TAU,
        help=f'Threshold in seconds for Ptau (default: {D.QUALITY_ASSESS_TAU})',
    )
    p_assess.add_argument(
        '--output_report',
        default=D.QUALITY_ASSESS_OUTPUT,
        help=f'Output CSV report filename (default: {D.QUALITY_ASSESS_OUTPUT})',
    )


    args = parser.parse_args(argv)

    if args.cmd in ('plot', 'map-subtitles', 'warp-video'):
        needs_artifacts = args.cmd != 'warp-video' or not args.curve_csv
        if needs_artifacts:
            try:
                _alignment_paths_from_args(args)
            except ValueError as e:
                parser.error(str(e))
    
    if args.cmd == 'assess-quality':
        try:
            io_utils.resolve_alignment_paths(
                session=args.ref_session,
                artifacts_dir=args.artifacts_dir,
            )
            io_utils.resolve_alignment_paths(
                session=args.test_session,
                artifacts_dir=args.artifacts_dir,
            )
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
    elif args.cmd == 'assess-quality':
        cmd_assess_quality(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
