#!/usr/bin/env python3
"""
generate_boards.py — Generate synthetic chess board PNG images for testing.
Requires only Python stdlib (no Pillow/numpy needed).

Usage:
    python3 scripts/generate_boards.py --count 200 --output data/boards_large
"""

import struct
import zlib
import os
import random
import argparse


def write_png(filename, pixels_rgb, width, height):
    """Write a PNG file using only Python stdlib."""
    def make_chunk(name, data):
        raw = name + data
        crc = zlib.crc32(raw) & 0xFFFFFFFF
        return struct.pack('>I', len(data)) + raw + struct.pack('>I', crc)

    sig  = b'\x89PNG\r\n\x1a\n'
    ihdr = make_chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    raw_rows = b''
    for y in range(height):
        raw_rows += b'\x00'
        raw_rows += bytes(pixels_rgb[y])
    idat = make_chunk(b'IDAT', zlib.compress(raw_rows, 6))
    iend = make_chunk(b'IEND', b'')

    with open(filename, 'wb') as f:
        f.write(sig + ihdr + idat + iend)


LIGHT  = (240, 217, 181)
DARK   = (181, 136,  99)
BLACK_PIECE = (40,  30,  25)
WHITE_PIECE = (245, 240, 228)


def draw_circle(pixels, cx, cy, radius, color, width, height):
    """Draw a filled circle (piece silhouette) onto the pixel grid."""
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                x, y = cx + dx, cy + dy
                if 0 <= x < width and 0 <= y < height:
                    pixels[y][x * 3:x * 3 + 3] = list(color)


def generate_board(seed, size=512):
    sq = size // 8
    pixels = []
    for py in range(size):
        row = []
        for px in range(size):
            r = py // sq
            f = px // sq
            color = LIGHT if (r + f) % 2 == 0 else DARK
            row.extend(color)
        pixels.append(row)

    rng = random.Random(seed)
    n_pieces = rng.randint(6, 24)
    positions = set()
    for _ in range(n_pieces):
        pr = rng.randint(0, 7)
        pf = rng.randint(0, 7)
        if (pr, pf) in positions:
            continue
        positions.add((pr, pf))
        cx = pf * sq + sq // 2
        cy = pr * sq + sq // 2
        radius = sq // 3
        color = BLACK_PIECE if rng.random() < 0.5 else WHITE_PIECE
        draw_circle(pixels, cx, cy, radius, color, size, size)

    return pixels, size, size


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic chess board PNGs')
    parser.add_argument('--count',  type=int, default=10,              help='Number of images')
    parser.add_argument('--size',   type=int, default=512,             help='Board size in pixels')
    parser.add_argument('--output', type=str, default='data/sample_boards', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f'Generating {args.count} chess board images ({args.size}×{args.size}) → {args.output}/')
    for i in range(args.count):
        pixels, w, h = generate_board(seed=i * 31 + 7, size=args.size)
        path = os.path.join(args.output, f'board_{i:03d}.png')
        write_png(path, pixels, w, h)
        if (i + 1) % 10 == 0 or i == args.count - 1:
            print(f'  {i + 1}/{args.count} done')

    print(f'Done. Files in {args.output}/')


if __name__ == '__main__':
    main()
