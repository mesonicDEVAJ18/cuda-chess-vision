#!/usr/bin/env python3
"""
visualize.py — plot intensity heatmaps and evaluation rankings
Usage: python3 scripts/visualize.py --results results/ --output plots/
Requires: matplotlib (pip install matplotlib)
"""
import argparse, os, glob, csv

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)

def plot_heatmap(rows, title, outpath):
    data = [[float(rows[r][c]) for c in 'abcdefgh'] for r in range(len(rows))]
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(data, cmap='plasma', vmin=0, vmax=255)
    ax.set_xticks(range(8)); ax.set_xticklabels(list('abcdefgh'))
    ax.set_yticks(range(8)); ax.set_yticklabels([str(r) for r in range(8,0,-1)])
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85, label='Mean intensity')
    plt.tight_layout()
    plt.savefig(outpath, dpi=120); plt.close()

def plot_evaluation(rows, outpath):
    names  = [r['filename'][:20] for r in rows[:10]]
    scores = [float(r['score']) for r in rows[:10]]
    pawns  = [float(r['pawn_fft_energy']) for r in rows[:10]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    bars = ax1.barh(names[::-1], scores[::-1], color='steelblue')
    ax1.set_xlabel('Evaluation Score'); ax1.set_title('Board Rankings (cuBLAS)')
    ax1.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)

    ax2.barh(names[::-1], pawns[::-1], color='coral')
    ax2.set_xlabel('FFT Energy'); ax2.set_title('Pawn Structure Complexity (cuFFT)')

    plt.tight_layout()
    plt.savefig(outpath, dpi=120); plt.close()
    print(f'  Saved: {outpath}')

def ascii_heatmap(rows):
    chars = ' .,:;+*#@'
    print("    a  b  c  d  e  f  g  h")
    for i, row in enumerate(rows):
        rank = 8-i
        line = f"  {rank} "
        for c in 'abcdefgh':
            v = float(row[c])
            line += chars[int(v/255*(len(chars)-1))]*2+' '
        print(line)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/')
    p.add_argument('--output',  default='plots/')
    a = p.parse_args()

    if not HAS_MPL:
        print("matplotlib not found — ASCII fallback\n")
        for csv_path in sorted(glob.glob(os.path.join(a.results,'intensity_*.csv')))[:3]:
            rows = read_csv(csv_path)
            print(f'\n{os.path.basename(csv_path)}')
            ascii_heatmap(rows)
        eval_path = os.path.join(a.results,'evaluation.csv')
        if os.path.exists(eval_path):
            rows = read_csv(eval_path)
            print('\nEvaluation rankings:')
            print(f"  {'Rank':<5} {'File':<30} {'Score':>8} {'PawnFFT':>10}")
            for r in rows[:10]:
                print(f"  {r['rank']:<5} {r['filename']:<30} {float(r['score']):>8.4f} {float(r['pawn_fft_energy']):>10.4f}")
        return

    os.makedirs(a.output, exist_ok=True)

    # Intensity heatmaps
    for csv_path in sorted(glob.glob(os.path.join(a.results,'intensity_*.csv'))):
        rows = read_csv(csv_path)
        name = os.path.splitext(os.path.basename(csv_path))[0]
        out  = os.path.join(a.output, f'heatmap_{name}.png')
        plot_heatmap(rows, name, out)
        print(f'  Saved: {out}')

    # Evaluation bar chart
    eval_path = os.path.join(a.results,'evaluation.csv')
    if os.path.exists(eval_path):
        rows = read_csv(eval_path)
        plot_evaluation(rows, os.path.join(a.output,'evaluation.png'))

    print(f'\nAll plots in: {a.output}')

if __name__=='__main__': main()
