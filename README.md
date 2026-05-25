# Opera alignment refactor

This repository contains audio-to-audio alignment using frame features and DTW. Supported feature extractors:

- **openl3** (default) — learned embeddings (slow, highest quality potential)
- **mfcc** — librosa MFCCs (fast, timbre-focused)
- **chroma** — librosa chroma CQT or STFT (fast, pitch-class / harmony-focused)

Usage examples (from repo root):

Align audio and map subtitles (OpenL3):
```
python -m opera_align align --ref_wav sources/1.wav --stream_wav sources/2.wav [--subtitles sources/subtitles.csv] --out_prefix session3  
```

MFCC + DTW (no OpenL3 required):
```
python -m opera_align align --feature mfcc --ref_wav sources/1.wav --stream_wav sources/2.wav --out_prefix session_mfcc --backend librosa
```

Chroma + DTW:
```
python -m opera_align align --feature chroma --chroma_type cqt --ref_wav sources/1.wav --stream_wav sources/2.wav --out_prefix session_chroma --backend librosa
```

**Full pipeline** (align + plot + warp video in one step):
```
python -m opera_align pipeline --ref_wav sources/ref.wav --stream_wav sources/stream.wav --video sources/stream.mp4 --session myrun --feature mfcc --backend librosa
```

Or use the standalone script:
```
python scripts/align_and_warp.py --ref-wav sources/ref.wav --stream-wav sources/stream.wav --video sources/stream.mp4 --feature mfcc --backend librosa
```

Plot alignment (paths inferred from `--session` used during align):
```
python -m opera_align plot --session session3 --output_png alignment.png
```

Warp video:
```
python -m opera_align warp-video --session session2 --input_video sources/2.mp4 --output_video 2_warped.mp4
```

Explicit paths still work and override `--session` for individual files, e.g. `--session session3 --path_ref custom/path_ref.npy`.

### Web UI

Same commands and defaults as the CLI (align, pipeline, map-subtitles, plot, warp-video):

```
pip install -r requirements.txt
python -m opera_align.web
```

Open http://127.0.0.1:5000/ — upload files or enter paths under the repo working directory. Outputs appear under `output/` with download links.

Optional environment variables:
- `OPERA_ALIGN_WORK_DIR` — working directory (default: current directory)
- `OPERA_ALIGN_UPLOAD_DIR` — uploaded file staging (default: `uploads/`)

Install requirements:
```
pip install -r requirements.txt
```

Note: `moviepy` requires FFmpeg. `openl3` may download model weights and is only needed for `--feature openl3`.

### Feature flags

| Flag | Description |
|------|-------------|
| `--feature` | `openl3`, `mfcc`, or `chroma` |
| `--hop_size` | Frame hop in seconds (default `0.1`; used by OpenL3 and as default librosa hop) |
| `--hop_length` | Librosa hop in samples for mfcc/chroma (default: `hop_size * sr`) |
| `--n_mfcc` | Number of MFCC coefficients (default `20`) |
| `--chroma_type` | `cqt` (default) or `stft` |
| `--backend` | DTW backend: `fastdtw`, `librosa`, or `fallback` |
