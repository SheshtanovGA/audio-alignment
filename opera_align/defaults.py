"""CLI and web UI defaults (single source of truth)."""

# Shared alignment / feature options (align + pipeline)
SR = 48000
FEATURE = "openl3"
EMBEDDING_SIZE = 512
HOP_SIZE = 0.1
HOP_LENGTH = None
N_MFCC = 20
CHROMA_TYPE = "cqt"
BACKEND = "fastdtw"

FEATURE_CHOICES = ("openl3", "mfcc", "chroma")
CHROMA_TYPE_CHOICES = ("cqt", "stft")
BACKEND_CHOICES = ("fastdtw", "librosa", "fallback")

# align
ALIGN_OUT_PREFIX = "artifacts/out"
ALIGN_OUT_SUBTITLES = "artifacts/mapped_subtitles.csv"

# pipeline
PIPELINE_OUTPUT_DIR = "output"
PIPELINE_ARTIFACTS_DIR = "artifacts"
PIPELINE_PLOT_NAME = "alignment.png"
PIPELINE_VIDEO_NAME = "warped.mp4"

# map-subtitles
MAP_OUTPUT_CSV = "mapped_subtitles.csv"

# plot
PLOT_OUTPUT_PNG = "alignment_plot.png"

# session / artifacts
ARTIFACTS_DIR = "artifacts"

ALIGNMENT_ARTIFACT_NAMES = ("ts_ref", "ts_stream", "path_ref", "path_stream")
