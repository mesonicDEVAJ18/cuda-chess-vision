# CUDA Chess Vision

GPU-accelerated chess board image processing and position evaluation pipeline using four CUDA GPU libraries working together: **NPP**, **cuBLAS**, **cuFFT**, and **Thrust**.

---

## What it does

Takes a directory of chess board PNG images and runs them through a two-stage GPU pipeline:

**Stage 1 — Image Processing (NPP + custom CUDA kernels)**
Each image goes through: RGB→grayscale → Gaussian blur → Sobel edge detection → gradient magnitude (custom kernel) → histogram equalization (custom kernels) → per-square intensity extraction (custom kernel). The result is a processed grayscale image and a 64-value intensity map — one float per chess square — representing how bright each square is after processing.

**Stage 2 — Board Evaluation (cuBLAS + cuFFT + Thrust)**
The 64-value intensity maps from all boards are evaluated in batch:
- **cuBLAS** (`cublasSgemm`) — multiplies the [N×64] intensity matrix by a [64×4] piece-square weight matrix, scoring each board across four features: material balance, king safety, pawn structure, and piece activity
- **cuFFT** (`cufftExecR2C`) — runs a real-to-complex FFT on the pawn row of each board; high-frequency energy indicates a fragmented pawn structure
- **Thrust** (`thrust::sort_by_key`) — sorts all boards by evaluation score in parallel and assigns ranks

---

## Project structure

```
cuda-chess-vision/
├── README.md
├── Makefile
├── run.sh
├── include/
│   ├── image_io.h          # PNG I/O interface
│   ├── pipeline.h          # NPP pipeline interface
│   └── evaluator.h         # cuBLAS/cuFFT/Thrust evaluator interface
├── src/
│   ├── main.cu             # Entry point, CLI, two-stage orchestration
│   ├── pipeline.cu         # Stage 1: NPP + 4 custom CUDA kernels
│   ├── evaluator.cu        # Stage 2: cuBLAS + cuFFT + Thrust
│   └── image_io.c          # PNG load/save (libpng), CSV, directory scan
├── scripts/
│   ├── generate_boards.py  # Generate synthetic board PNGs (stdlib only)
│   └── visualize.py        # Plot heatmaps and evaluation rankings
├── data/
│   └── sample_boards/      # 10 pre-generated 512×512 board PNGs
└── output_samples/         # Example outputs committed to repo
    ├── execution_log.txt
    ├── evaluation.csv
    └── intensity_board_000.csv
```

---

## Requirements

- NVIDIA GPU (Compute Capability 6.0+; tested on T4 = sm_75)
- CUDA Toolkit 11.x or 12.x
- libpng development headers

```bash
sudo apt-get install libpng-dev build-essential
```

### Check your environment

```bash
nvcc --version
nvidia-smi
ls /usr/local/cuda/lib64/libnpp* /usr/local/cuda/lib64/libcublas* /usr/local/cuda/lib64/libcufft*
```

---

## Build

```bash
make
```

```bash
sudo apt-get install gcc-12   # preferred
# OR just run make — it falls back automatically with -allow-unsupported-compiler
```

If CUDA is not at `/usr/local/cuda`:

```bash
make CUDA_PATH=/usr/local/cuda-12.x
```

---

## Run

```bash
make run
# equivalent to:
./chess_vision --input data/sample_boards --output results --csv --verbose
```

### Full CLI

```bash
./chess_vision --input <dir> [OPTIONS]

  --input   <dir>   PNG image directory        (required)
  --output  <dir>   Output directory            (default: results)
  --batch   <N>     Images per batch            (default: 32)
  --csv             Export CSVs
  --verbose         Print per-image timing
  --help
```

### Generate a larger dataset

```bash
python3 scripts/generate_boards.py --count 200 --output data/boards_200
./chess_vision --input data/boards_200 --output results --csv --verbose
```

---

## GPU pipeline in detail

```
Input PNG
  │
  ├─[CPU]    libpng decode → RGB host buffer
  ├─[CPU]    cudaMemcpy H→D
  │
  ├─[NPP]    nppiRGBToGray_8u_C3C1R          RGB → grayscale
  ├─[NPP]    nppiFilterGauss_8u_C1R          3×3 Gaussian blur
  ├─[NPP]    nppiFilterSobelHoriz_8u_C1R     horizontal Sobel Gx
  ├─[NPP]    nppiFilterSobelVert_8u_C1R      vertical Sobel Gy
  ├─[kernel] k_combine_edges                 √(Gx²+Gy²) magnitude
  ├─[kernel] k_histogram                     256-bin shared-mem histogram
  ├─[kernel] k_apply_lut                     histogram equalization LUT
  ├─[kernel] k_square_intensities            8×8 shared-mem reduction
  │
  ├─[CPU]    cudaMemcpy D→H → save PNG + CSV
  │
  └─ (repeat for all N boards, then:)
  │
  ├─[cuBLAS] cublasSgemm   [Nx64] × [64x4] → [Nx4] feature scores
  ├─[kernel] k_extract_pawn_row              extract pawn row signal
  ├─[cuFFT]  cufftExecR2C  batch R2C FFT on pawn rows
  ├─[kernel] k_fft_energy                   high-freq bin energy
  ├─[kernel] k_aggregate_scores             dot [Nx4] with weights → [N]
  └─[Thrust] sort_by_key   sort boards by score, assign ranks
```

---

## Output files

For each input `board_NNN.png`:
- `results/proc_board_NNN.png` — processed grayscale image
- `results/intensity_board_NNN.csv` — 8×8 per-square intensity table

For the full batch:
- `results/evaluation.csv` — ranked evaluation table with cuBLAS score and cuFFT pawn energy per board

### evaluation.csv format

```
rank,filename,score,material_proxy,pawn_fft_energy
1,board_004.png,0.4821,0.9642,12.334
2,board_007.png,0.4613,0.9226,14.112
...
```

---

## Performance (NVIDIA T4, 512×512 images)

| Stage          | 10 images | 200 images |
|----------------|-----------|------------|
| NPP pipeline   | ~28 ms    | ~560 ms    |
| cuBLAS eval    | <1 ms     | <2 ms      |
| cuFFT analysis | <1 ms     | <1 ms      |
| Thrust sort    | <1 ms     | <1 ms      |
| **Total**      | **~0.09 s** | **~0.58 s** |

---

## Lessons learned

- **NPP API stability**: `nppiEqualizeHist_8u_C1R` was removed in newer NPP versions — replaced with three portable custom kernels (histogram → LUT → apply) that work across all CUDA versions
- **cuBLAS column-major**: cuBLAS uses Fortran/column-major layout; getting the `cublasSgemm` transpose arguments right to compute row-major [Nx64]×[64x4] requires transposing both operands mentally
- **cuFFT batch mode**: `cufftPlanMany` with stride/distance parameters cleanly handles N independent 8-point FFTs in a single kernel call — much faster than N separate `cufftPlan1d` + execute calls
- **Thrust with raw pointers**: `thrust::device_vector` wrapping raw `cudaMalloc` pointers via iterator construction avoids double-allocation while keeping Thrust's clean sort API
- **gcc/nvcc version mismatch**: nvcc has a hard upper bound on supported gcc versions; the Makefile auto-detects gcc-12 and falls back to `-allow-unsupported-compiler` to avoid blocking the build on newer Linux distros

---

## License

MIT
