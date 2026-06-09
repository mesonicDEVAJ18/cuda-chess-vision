#!/usr/bin/env bash
# run.sh — CUDA Chess Vision full pipeline
# Usage: ./run.sh [--boards N] [--quality Q] [--large]
set -e

BOARDS=20
QUALITY=50
LARGE=0

for arg in "$@"; do
  case $arg in
    --boards)  shift; BOARDS=$1  ;;
    --quality) shift; QUALITY=$1 ;;
    --large)   BOARDS=100        ;;
  esac
  shift 2>/dev/null || true
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║        CUDA Chess Vision  —  run.sh              ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Auto-find nvcc if not in PATH
if ! which nvcc &>/dev/null; then
  for d in /usr/local/cuda*/bin /usr/cuda/bin /opt/cuda/bin; do
    if [ -x "$d/nvcc" ]; then
      export PATH="$d:$PATH"
      export CUDA_PATH="$(dirname $d)"
      echo "  Found CUDA at: $CUDA_PATH"
      break
    fi
  done
fi

echo "  Building…"
make all 2>&1 | tail -5
echo ""

echo "  Running pipeline (boards=$BOARDS, quality=$QUALITY)…"
./chess_vision \
    --boards  "$BOARDS"  \
    --quality "$QUALITY" \
    --output  results    \
    --verbose

echo ""
echo "  Generating plots…"
python3 scripts/visualize.py --results results --output plots || \
  echo "  (Install matplotlib for plots: pip install matplotlib numpy)"

echo ""
echo "  Output:"
echo "    results/boards/       — rendered board PNGs"
echo "    results/compressed/   — DCT-compressed board PNGs"
echo "    results/evaluation.csv"
echo "    results/compression.csv"
echo "    plots/                — matplotlib figures"
echo ""
