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
ALIGN_OUT_PREFIX = "cache/out"
ALIGN_OUT_SUBTITLES = "cache/mapped_subtitles.csv"

# pipeline
PIPELINE_OUTPUT_DIR = "output"
PIPELINE_CACHE_DIR = "cache"
PIPELINE_PLOT_NAME = "alignment.png"
PIPELINE_VIDEO_NAME = "warped.mp4"

# map-subtitles
MAP_OUTPUT_CSV = "mapped_subtitles.csv"

# plot
PLOT_OUTPUT_PNG = "alignment_plot.png"

# session / cache
CACHE_DIR = "cache"

ALIGNMENT_ARTIFACT_NAMES = ("ts_ref", "ts_stream", "path_ref", "path_stream")

# quality assessment
QUALITY_ASSESS_TAU = 0.1  # Threshold in seconds for Ptau
QUALITY_ASSESS_OUTPUT = "quality_assessment.csv"
