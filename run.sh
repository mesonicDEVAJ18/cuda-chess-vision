#!/usr/bin/env bash
# run.sh — build and run CUDA Chess Vision
set -e

echo "================================================"
echo "  CUDA Chess Vision — Capstone Edition"
echo "  GPU libs: NPP | cuBLAS | cuFFT | Thrust"
echo "================================================"
echo ""

# Auto-find nvcc if not in PATH
if ! which nvcc &>/dev/null; then
  for d in /usr/local/cuda*/bin /usr/cuda/bin /opt/cuda/bin; do
    if [ -x "$d/nvcc" ]; then
      export PATH="$d:$PATH"
      export CUDA_PATH="$(dirname $d)"
      echo "Found CUDA at: $CUDA_PATH"
      break
    fi
  done
fi

make all

echo ""
mkdir -p results
./chess_vision \
    --input   data/boards_200 \
    --output  results \
    --batch   32 \
    --csv \
    --verbose

echo ""
echo "--- Output files ---"
ls results/
