#!/usr/bin/env python3
"""
visualize.py — CUDA Chess Vision post-processing visualiser

Reads from results/ and produces 4 plots in plots/:
  1. board_gallery.png      — grid of rendered board images
  2. compression.png        — original vs compressed side-by-side + PSNR
  3. evaluation.png         — bar chart of evaluation scores (top boards)
  4. score_breakdown.png    — stacked bar: material / positional / pawn structure

Usage:
    python3 scripts/visualize.py --results results --output plots

Requires: matplotlib, numpy  (pip install matplotlib numpy)
Falls back to ASCII summary if matplotlib is not available.
"""

import argparse
import os
import glob
import csv
import sys

# ── dependency check ─────────────────────────────────────────────────────────
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")          # headless backend (no display needed)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── helpers ───────────────────────────────────────────────────────────────────

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def short(name, n=18):
    """Shorten a filename for axis labels."""
    return name[:n] + "…" if len(name) > n else name


# ── Plot 1 — Board Gallery ────────────────────────────────────────────────────

def plot_board_gallery(board_dir, out_path, max_boards=16):
    pngs = sorted(glob.glob(os.path.join(board_dir, "*.png")))[:max_boards]
    if not pngs:
        print("  [viz] No board PNGs found, skipping gallery.")
        return

    n = len(pngs)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3, rows * 3 + 0.6),
                             facecolor="#1a1a2e")
    fig.suptitle("Generated Chess Boards  (Stage 1 — cuRAND)",
                 color="white", fontsize=14, fontweight="bold", y=1.01)

    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    for idx, path in enumerate(pngs):
        try:
            img = plt.imread(path)
        except Exception:
            continue
        ax = axes[idx]
        ax.imshow(img)
        ax.set_title(os.path.basename(path).replace(".png",""),
                     color="#a0c4ff", fontsize=7, pad=3)
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close()
    print(f"  [viz] Saved: {out_path}")


# ── Plot 2 — Compression Comparison ──────────────────────────────────────────

def plot_compression(board_dir, comp_dir, comp_csv_path, out_path, max_show=4):
    rows_csv = read_csv(comp_csv_path)
    board_pngs = sorted(glob.glob(os.path.join(board_dir, "*.png")))[:max_show]
    if not board_pngs:
        print("  [viz] No boards for compression plot, skipping.")
        return

    n = len(board_pngs)
    fig, axes = plt.subplots(n, 2,
                             figsize=(8, n * 3 + 0.8),
                             facecolor="#1a1a2e")
    if n == 1:
        axes = [axes]

    fig.suptitle("DCT Compression  (Stage 2 — NPP + Block-DCT kernels)",
                 color="white", fontsize=13, fontweight="bold")

    # Build a PSNR lookup dict
    psnr_map = {}
    ratio_map = {}
    for r in rows_csv:
        psnr_map[r["filename"]] = float(r["psnr_db"])
        ratio_map[r["filename"]] = float(r["nonzero_coeff_ratio"])

    for i, bp in enumerate(board_pngs):
        bn = os.path.basename(bp)
        stem = bn.replace(".png","")

        # original
        try:
            orig = plt.imread(bp)
        except Exception:
            continue
        ax0 = axes[i][0]
        ax0.imshow(orig)
        ax0.set_title("Original", color="#a0c4ff", fontsize=9)
        ax0.axis("off")

        # compressed
        cp = os.path.join(comp_dir, bn)
        ax1 = axes[i][1]
        if os.path.exists(cp):
            try:
                comp = plt.imread(cp)
                ax1.imshow(comp, cmap="gray")
                psnr  = psnr_map.get(bn, float("nan"))
                ratio = ratio_map.get(bn, float("nan"))
                ax1.set_title(
                    f"Compressed  PSNR={psnr:.1f} dB  density={ratio*100:.0f}%",
                    color="#ffd6a5", fontsize=8)
            except Exception:
                ax1.text(0.5,0.5,"Error loading",ha="center",
                         color="red",transform=ax1.transAxes)
        else:
            ax1.text(0.5,0.5,"Not found",ha="center",
                     color="gray",transform=ax1.transAxes)
        ax1.axis("off")

        # row label
        axes[i][0].set_ylabel(stem, color="white", fontsize=8, rotation=0,
                              labelpad=40, va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close()
    print(f"  [viz] Saved: {out_path}")


# ── Plot 3 — Evaluation Rankings ─────────────────────────────────────────────

def plot_evaluation(eval_csv_path, out_path, top_n=15):
    rows = read_csv(eval_csv_path)
    if not rows:
        print("  [viz] No evaluation CSV, skipping evaluation plot.")
        return

    rows = sorted(rows, key=lambda r: int(r["rank"]))[:top_n]
    names  = [short(r["filename"]) for r in rows]
    scores = [float(r["score"]) for r in rows]

    # colour gradient: gold → silver → bronze → blue
    colours = []
    for i in range(len(rows)):
        if   i == 0: colours.append("#FFD700")
        elif i == 1: colours.append("#C0C0C0")
        elif i == 2: colours.append("#CD7F32")
        else:        colours.append("#4a9eff")

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows)*0.55 + 1)),
                           facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.barh(names[::-1], scores[::-1], color=colours[::-1],
                   edgecolor="#ffffff22", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=4, color="white", fontsize=8)

    ax.set_xlabel("Evaluation Score (cuBLAS + PST + cuFFT)", color="#a0c4ff")
    ax.set_title("Board Rankings  (Stage 3 — cuBLAS + cuFFT + Thrust)",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")
    ax.xaxis.label.set_color("#a0c4ff")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close()
    print(f"  [viz] Saved: {out_path}")


# ── Plot 4 — Score Breakdown ──────────────────────────────────────────────────

def plot_score_breakdown(eval_csv_path, out_path, top_n=12):
    rows = read_csv(eval_csv_path)
    if not rows:
        print("  [viz] No evaluation CSV, skipping breakdown plot.")
        return

    rows = sorted(rows, key=lambda r: int(r["rank"]))[:top_n]
    names     = [short(r["filename"]) for r in rows]
    material  = [float(r["material"])       for r in rows]
    positional= [float(r["positional"])     for r in rows]
    pawn_fft  = [float(r["pawn_fft_energy"])for r in rows]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(rows)*0.9), 5),
                           facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.bar(x - w,  material,   w, label="Material (cuBLAS)",   color="#4a9eff", alpha=0.9)
    ax.bar(x,      positional, w, label="Positional PST",       color="#06d6a0", alpha=0.9)
    ax.bar(x + w, [-v*0.3 for v in pawn_fft], w,
           label="Pawn Structure FFT (penalty)", color="#ef476f", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", color="white", fontsize=8)
    ax.set_ylabel("Score component", color="#a0c4ff")
    ax.set_title("Evaluation Breakdown by Component",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")
    ax.axhline(0, color="#ffffff44", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close()
    print(f"  [viz] Saved: {out_path}")


# ── ASCII fallback ────────────────────────────────────────────────────────────

def ascii_summary(results_dir):
    eval_rows = read_csv(os.path.join(results_dir, "evaluation.csv"))
    comp_rows = read_csv(os.path.join(results_dir, "compression.csv"))

    print("\n── Evaluation Rankings ─────────────────────────────────")
    print(f"  {'Rank':<5} {'File':<22} {'Score':>7} {'Material':>9} {'PST':>7} {'PawnFFT':>8}")
    for r in sorted(eval_rows, key=lambda x: int(x["rank"]))[:10]:
        print(f"  {r['rank']:<5} {r['filename']:<22} "
              f"{float(r['score']):>7.3f} {float(r['material']):>9.3f} "
              f"{float(r['positional']):>7.3f} {float(r['pawn_fft_energy']):>8.3f}")

    if comp_rows:
        avg_psnr = sum(float(r["psnr_db"]) for r in comp_rows) / len(comp_rows)
        avg_dens = sum(float(r["nonzero_coeff_ratio"]) for r in comp_rows) / len(comp_rows)
        print(f"\n── Compression Stats ────────────────────────────────────")
        print(f"  Average PSNR            : {avg_psnr:.2f} dB")
        print(f"  Average coeff density   : {avg_dens*100:.1f}%")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CUDA Chess Vision visualiser")
    p.add_argument("--results", default="results", help="results directory")
    p.add_argument("--output",  default="plots",   help="output directory for plots")
    a = p.parse_args()

    board_dir   = os.path.join(a.results, "boards")
    comp_dir    = os.path.join(a.results, "compressed")
    eval_csv    = os.path.join(a.results, "evaluation.csv")
    comp_csv    = os.path.join(a.results, "compression.csv")

    if not HAS_MPL:
        print("[viz] matplotlib not found — ASCII fallback\n")
        ascii_summary(a.results)
        return

    os.makedirs(a.output, exist_ok=True)
    print(f"\n[viz] Reading from : {a.results}/")
    print(f"[viz] Writing to   : {a.output}/\n")

    plot_board_gallery(board_dir,
                       os.path.join(a.output, "board_gallery.png"))

    plot_compression(board_dir, comp_dir, comp_csv,
                     os.path.join(a.output, "compression.png"))

    plot_evaluation(eval_csv,
                    os.path.join(a.output, "evaluation.png"))

    plot_score_breakdown(eval_csv,
                         os.path.join(a.output, "score_breakdown.png"))

    print(f"\n[viz] All plots in: {a.output}/\n")


if __name__ == "__main__":
    main()
