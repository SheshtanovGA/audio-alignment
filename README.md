# Opera alignment refactor

This repository contains a refactored skeleton for audio-to-audio alignment using OpenL3 embeddings and DTW.

Usage examples (from repo root):

Align audio and map subtitles:
```
python -m opera_align align --ref_wav sources/1.wav --stream_wav sources/2.wav [--subtitles sources/subtitles.csv] --out_prefix session3  
```

Plot alignment:
```
python -m opera_align plot --ts_ref artifacts/session3/ts_ref.npy --ts_stream artifacts/session3/ts_stream.npy --path_ref artifacts/session3/path_ref.npy --path_stream artifacts/session3/path_stream.npy --output_png alignment.png
```

Warp video:
```
python -m opera_align warp-video --input_video sources/2.mp4 --output_video 2_warped.mp4 --ts_ref artifacts/session2/ts_ref.npy --ts_stream artifacts/session2/ts_stream.npy --path_ref artifacts/session2/path_ref.npy --path_stream artifacts/session2/path_stream.npy                                        
```

Install requirements:
```
pip install -r requirements.txt
```

Note: `moviepy` requires FFmpeg. `openl3` may download model weights.
