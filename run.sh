#!/usr/bin/env bash
# run.sh — build and run CUDA Chess Vision on sample data
set -e

echo "================================================"
echo "  CUDA Chess Vision — GPU Board Image Pipeline"
echo "================================================"
echo ""

make all

echo ""
echo "Running on data/sample_boards/ ..."
mkdir -p results

./chess_vision \
    --input   data/sample_boards \
    --output  results \
    --batch   32 \
    --csv \
    --verbose

echo ""
echo "Processed images:"
ls results/proc_*.png 2>/dev/null | head -10

echo ""
echo "Intensity CSVs:"
ls results/intensity_*.csv 2>/dev/null | head -5

echo ""
echo "Sample CSV (board_000):"
cat results/intensity_board_000.csv 2>/dev/null || \
    cat "$(ls results/intensity_*.csv 2>/dev/null | head -1)" 2>/dev/null || \
    echo "(no CSV found)"
