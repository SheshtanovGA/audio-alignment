"""Simple CLI to run alignment tasks."""
import argparse
import numpy as np
import os

from . import io_utils, audio_features, alignment, subtitle_mapper, plotting, video_warp


def cmd_align(args):
    import librosa
    try:
        import openl3
    except Exception as e:
        raise RuntimeError("openl3 is required for embedding extraction") from e

    print("Loading audio...")
    y_ref, sr_ref = audio_features.load_audio(librosa, args.ref_wav, sr=args.sr)
    y_stream, sr_stream = audio_features.load_audio(librosa, args.stream_wav, sr=args.sr)

    print("Extracting embeddings (this may take a while)...")
    emb_ref, ts_ref = audio_features.extract_openl3_embeddings(openl3, y_ref, sr_ref, embedding_size=args.embedding_size, hop_size=args.hop_size)
    emb_stream, ts_stream = audio_features.extract_openl3_embeddings(openl3, y_stream, sr_stream, embedding_size=args.embedding_size, hop_size=args.hop_size)

    emb_ref = audio_features.normalize_embeddings(emb_ref)
    emb_stream = audio_features.normalize_embeddings(emb_stream)

    print("Computing DTW...")
    path_ref, path_stream = alignment.compute_dtw_path(emb_ref, emb_stream, backend=args.backend)

    mapped_ref_times = ts_ref[path_ref]
    mapped_stream_times = ts_stream[path_stream]

    io_utils.safe_npy_save(ts_ref, "artifacts/" + args.out_prefix + "/" + "ts_ref.npy")
    io_utils.safe_npy_save(ts_stream, "artifacts/" + args.out_prefix + "/" + "ts_stream.npy")
    io_utils.safe_npy_save(path_ref, "artifacts/" + args.out_prefix + "/" + "path_ref.npy")
    io_utils.safe_npy_save(path_stream, "artifacts/" + args.out_prefix + "/" + "path_stream.npy")

    if args.subtitles:
        subtitle_mapper.map_subtitles(args.subtitles, mapped_ref_times, mapped_stream_times, "output/" + args.out_subtitles)
        print(f"Mapped subtitles written to output/{args.out_subtitles}")


def cmd_map_subtitles(args):
    ts_ref = io_utils.safe_npy_load(args.ts_ref)
    ts_stream = io_utils.safe_npy_load(args.ts_stream)
    path_ref = io_utils.safe_npy_load(args.path_ref)
    path_stream = io_utils.safe_npy_load(args.path_stream)

    mapped_ref_times = ts_ref[path_ref]
    mapped_stream_times = ts_stream[path_stream]
    subtitle_mapper.map_subtitles(args.subtitles_csv, mapped_ref_times, mapped_stream_times, "output/" + args.output_csv)
    print(f"Mapped subtitles written to output/{args.output_csv}")


def cmd_plot(args):
    ts_ref = io_utils.safe_npy_load(args.ts_ref)
    ts_stream = io_utils.safe_npy_load(args.ts_stream)
    path_ref = io_utils.safe_npy_load(args.path_ref)
    path_stream = io_utils.safe_npy_load(args.path_stream)
    plotting.plot_alignment(ts_ref, ts_stream, path_ref, path_stream, args.subtitles_csv, "output/" + args.output_png)
    print(f"Plot written to output/{args.output_png}")

def cmd_warp(args):
    import pandas as pd
    if args.curve_csv:
        df = pd.read_csv(args.curve_csv)
        mapped_ref_times = df['reference_time'].values
        mapped_stream_times = df['stream_time'].values
    else:
        ts_ref = io_utils.safe_npy_load(args.ts_ref)
        ts_stream = io_utils.safe_npy_load(args.ts_stream)
        path_ref = io_utils.safe_npy_load(args.path_ref)
        path_stream = io_utils.safe_npy_load(args.path_stream)
        mapped_ref_times = ts_ref[path_ref]
        mapped_stream_times = ts_stream[path_stream]

    video_warp.warp_video(args.input_video, mapped_ref_times, mapped_stream_times, "output/" + args.output_video)
    print(f"Warped video written to output/{args.output_video}")


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
    p_align.add_argument('--embedding_size', type=int, default=512)
    p_align.add_argument('--hop_size', type=float, default=0.1)
    p_align.add_argument('--backend', choices=['fastdtw', 'librosa', 'fallback'], default='fastdtw')

    p_map = sub.add_parser('map-subtitles')
    p_map.add_argument('--ts_ref', required=True)
    p_map.add_argument('--ts_stream', required=True)
    p_map.add_argument('--path_ref', required=True)
    p_map.add_argument('--path_stream', required=True)
    p_map.add_argument('--subtitles_csv', required=True)
    p_map.add_argument('--output_csv', default='mapped_subtitles.csv')

    p_plot = sub.add_parser('plot')
    p_plot.add_argument('--ts_ref', required=True)
    p_plot.add_argument('--ts_stream', required=True)
    p_plot.add_argument('--path_ref', required=True)
    p_plot.add_argument('--path_stream', required=True)
    p_plot.add_argument('--subtitles_csv', required=False)
    p_plot.add_argument('--output_png', default='alignment_plot.png')

    p_warp = sub.add_parser('warp-video')
    p_warp.add_argument('--input_video', required=True)
    p_warp.add_argument('--output_video', required=True)
    p_warp.add_argument('--curve_csv', required=False)
    p_warp.add_argument('--ts_ref', required=False)
    p_warp.add_argument('--ts_stream', required=False)
    p_warp.add_argument('--path_ref', required=False)
    p_warp.add_argument('--path_stream', required=False)

    args = parser.parse_args(argv)
    if args.cmd == 'align':
        cmd_align(args)
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
