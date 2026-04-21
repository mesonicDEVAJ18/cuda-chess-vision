#!/usr/bin/env python3
"""
visualize.py — Plot intensity heatmaps and side-by-side comparisons.

Usage:
    python3 scripts/visualize.py --results results/ --input data/sample_boards/ --output plots/
"""

import argparse
import os
import glob
import struct
import zlib

try:
    import numpy as np
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def read_png_gray(path):
    """Read a grayscale PNG using only stdlib (for environments without Pillow)."""
    with open(path, 'rb') as f:
        sig = f.read(8)
        assert sig == b'\x89PNG\r\n\x1a\n', "Not a PNG"
        data_chunks = b''
        width = height = 0
        while True:
            length = struct.unpack('>I', f.read(4))[0]
            ctype  = f.read(4)
            chunk  = f.read(length)
            f.read(4)  # CRC
            if ctype == b'IHDR':
                width, height = struct.unpack('>II', chunk[:8])
            elif ctype == b'IDAT':
                data_chunks += chunk
            elif ctype == b'IEND':
                break
        raw = zlib.decompress(data_chunks)
        stride = width + 1  # 1 filter byte per row
        pixels = []
        for y in range(height):
            row = list(raw[y * stride + 1: y * stride + 1 + width])
            pixels.append(row)
        return pixels, width, height


def read_csv_intensities(path):
    with open(path) as f:
        lines = f.readlines()
    grid = []
    for line in lines[1:]:  # skip header
        parts = line.strip().split(',')
        grid.append([float(x) for x in parts[1:]])
    return grid


def plot_heatmap_ascii(grid, label):
    """ASCII fallback heatmap when matplotlib not available."""
    chars = ' .,:;+*#@'
    print(f"\n  Intensity heatmap: {label}")
    print("    a  b  c  d  e  f  g  h")
    for r, row in enumerate(grid):
        rank = 8 - r
        line = f"  {rank} "
        for v in row:
            idx = int(v / 255 * (len(chars) - 1))
            line += chars[idx] * 2 + ' '
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results/')
    parser.add_argument('--input',   default='data/sample_boards/')
    parser.add_argument('--output',  default='plots/')
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.results, 'intensity_*.csv')))
    print(f"Found {len(csv_files)} CSV files")

    if not HAS_MPL:
        print("matplotlib not available — showing ASCII heatmaps\n")
        for csv in csv_files[:5]:
            grid = read_csv_intensities(csv)
            plot_heatmap_ascii(grid, os.path.basename(csv))
        return

    os.makedirs(args.output, exist_ok=True)

    for csv in csv_files:
        grid = read_csv_intensities(csv)
        arr  = np.array(grid)
        name = os.path.splitext(os.path.basename(csv))[0]

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(arr, cmap='plasma', vmin=0, vmax=255, aspect='equal')
        ax.set_xticks(range(8)); ax.set_xticklabels(list('abcdefgh'))
        ax.set_yticks(range(8)); ax.set_yticklabels([str(r) for r in range(8, 0, -1)])
        ax.set_title(name, fontsize=10)
        plt.colorbar(im, ax=ax, label='Mean intensity (0–255)', shrink=0.85)
        plt.tight_layout()
        out = os.path.join(args.output, f'heatmap_{name}.png')
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  Saved: {out}")

    print(f"\nAll plots in: {args.output}")


if __name__ == '__main__':
    main()
