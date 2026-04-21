# CUDA Chess Vision

GPU-accelerated batch image processing pipeline for chess board images, built with CUDA and NVIDIA NPP (Performance Primitives).

Processes hundreds of chess board PNG images through a multi-stage GPU pipeline: RGB→grayscale conversion, Gaussian blur, Sobel edge detection, histogram equalization, and per-square intensity extraction. Each board's 8×8 grid of squares is analyzed and exported as a CSV signal.

---

## Project Structure

```
cuda-chess-vision/
├── README.md
├── Makefile
├── run.sh
├── include/
│   ├── image_io.h          # PNG I/O interface (libpng)
│   └── pipeline.h          # GPU pipeline interface
├── src/
│   ├── main.cu             # Entry point, CLI, batch loop
│   ├── pipeline.cu         # GPU pipeline (NPP + custom kernels)
│   └── image_io.c          # PNG load/save via libpng, CSV export
├── scripts/
│   ├── generate_boards.py  # Generate synthetic board images (stdlib only)
│   └── visualize.py        # Plot intensity heatmaps (requires matplotlib)
├── data/
│   └── sample_boards/      # 10 pre-generated 512×512 board PNGs
└── output_samples/         # Example outputs committed to repo
    ├── execution_log.txt
    └── intensity_board_000.csv
```

---

## Requirements

- NVIDIA GPU (Compute Capability 6.0+; tested on T4 = sm_75)
- CUDA Toolkit 11.x or 12.x (provides `nvcc`, `libnppc`, `libnppi`)
- libpng development headers: `sudo apt-get install libpng-dev` (Linux) 
  Windows: 
  git clone https://github.com/microsoft/vcpkg
  cd vcpkg
  .\bootstrap-vcpkg.bat

  .\vcpkg install libpng:x64-windows
  .\vcpkg integrate install
- GCC 9+

### Check your environment

```bash
nvcc --version
nvidia-smi
ls /usr/local/cuda/lib64/libnpp*
dpkg -l libpng-dev
```

---

## Build

```bash
make
```

Debug build (with `-G -g`):

```bash
make debug
```

Clean:

```bash
make clean
```

---

## Run

### Quick start (sample data included)

```bash
make run
# or equivalently:
bash run.sh
```

### Full CLI

```bash
./chess_vision --input <dir> [OPTIONS]

Options:
  --input   <dir>   Directory of input PNG images  (required)
  --output  <dir>   Output directory               (default: results)
  --batch   <N>     Images per batch               (default: 32)
  --csv             Export per-square intensity CSVs
  --verbose         Print per-image GPU timing
  --help
```

### Examples

```bash
# Process sample boards, export CSVs, print timing
./chess_vision --input data/sample_boards --output results --csv --verbose

# Generate 200 boards and process them
python3 scripts/generate_boards.py --count 200 --output data/boards_200
./chess_vision --input data/boards_200 --output results --batch 32 --csv --verbose

# Visualize intensity heatmaps (requires matplotlib)
python3 scripts/visualize.py --results results/ --input data/sample_boards/ --output plots/
```

---

## GPU Pipeline

```
Input PNG
  │
  ├─[CPU] libpng decode → RGB host buffer
  ├─[CPU] cudaMemcpy H→D (pinned staging)
  │
  ├─[NPP] nppiRGBToGray_8u_C3C1R          (RGB → grayscale)
  ├─[NPP] nppiFilterGauss_8u_C1R          (3×3 Gaussian blur)
  ├─[NPP] nppiFilterSobelHoriz_8u_C1R     (horizontal Sobel Gx)
  ├─[NPP] nppiFilterSobelVert_8u_C1R      (vertical Sobel Gy)
  ├─[CUDA kernel] k_combine_edges          (√(Gx²+Gy²), clamped)
  ├─[NPP] nppiEqualizeHist_8u_C1R         (histogram equalization)
  ├─[CUDA kernel] k_square_intensities     (8×8 shared-mem reduction)
  │
  └─[CPU] cudaMemcpy D→H → PNG + CSV save
```

### Custom CUDA Kernels

**`k_combine_edges`** — combines horizontal and vertical Sobel maps into gradient magnitude. Launched with 16×16 thread blocks covering the full image.

**`k_square_intensities`** — divides the board into an 8×8 grid and computes average pixel intensity per square using shared memory parallel reduction. Launched with a grid of 8×8 blocks (one per chess square) and 16×16 threads per block.

---

## Output

For each input `board_NNN.png`, the pipeline writes:

- `results/proc_board_NNN.png` — processed grayscale image (equalized edge map)
- `results/intensity_board_NNN.csv` — 8×8 per-square intensity table

### CSV format

```
rank,a,b,c,d,e,f,g,h
8,42.17,198.83,41.92,...
7,...
...
1,...
```

Values are mean pixel intensities (0–255) per chess square after equalization. High values correspond to light squares; lower values indicate piece-occupied or dark squares.

---

## Performance (NVIDIA T4, 512×512 images)

| Images | Total time | Throughput |
|--------|-----------|------------|
| 10     | ~0.06 s   | ~167 img/s |
| 100    | ~0.18 s   | ~556 img/s |
| 200    | ~0.31 s   | ~645 img/s |

Timings exclude one-time CUDA context initialization (~150–300 ms).

---

## Lessons Learned

- **NPP border handling**: `nppiFilterGauss` and `nppiFilterSobel*` do not support in-place operation; separate input/output buffers are required.
- **Pinned memory**: Using `cudaMallocHost` for staging buffers gives ~40% faster H↔D transfers vs pageable memory; for this project we kept it simple with regular malloc since transfer time is small relative to processing.
- **Shared memory reduction**: The `k_square_intensities` kernel allocates 256 floats (1 KB) per block — well within the 48 KB shared memory limit, allowing all 64 blocks to run simultaneously on a T4.
- **Batch size**: Processing 32 images per CPU loop iteration balances memory footprint and amortizes CUDA launch overhead. The GPU itself processes each image sequentially within the batch; a more advanced version would use CUDA streams to overlap H→D transfer of the next image with processing of the current one.

---

## Dataset

The 10 sample boards in `data/sample_boards/` were generated by `scripts/generate_boards.py` using only Python stdlib (no Pillow required). Each is a 512×512 PNG with randomized piece placement.

To generate a larger dataset:

```bash
python3 scripts/generate_boards.py --count 200 --size 512 --output data/boards_200
```

Real chess board images can also be used — any PNG files work as input.

---

## License

MIT
