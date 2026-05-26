#!/bin/bash
# Commit script for quality assessment implementation

cd "$(dirname "$0")"

# Stage all changes
git add -A

# Create commit with message following repository convention
git commit -m "add alignment quality assessment (ATE/Ptau metrics)" << 'EOF'
Implement assess-quality CLI command and web UI for quantitative alignment quality evaluation. Features:

- New quality_metrics module with ATE (Average Temporal Error) and Ptau (Proportion within tau) calculations
- assess-quality CLI command comparing reference and test sessions
- DTW path interpolation for control point mapping
- Web UI integration with form handling
- Comprehensive unit and integration tests
- Full documentation with usage examples

The command supports custom control points, configurable thresholds, and outputs detailed CSV reports with per-point errors and summary metrics.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
EOF

echo "Commit completed successfully!"
git log --oneline -1
