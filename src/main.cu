// main.cu — CUDA Chess Vision entry point
//
// Three-stage GPU pipeline:
//
//   Stage 1 — Board Generation  (board_gen.cu / cuRAND)
//     Generate N random legal chess positions on the GPU.
//     Render each board to a color PNG in results/boards/.
//
//   Stage 2 — DCT Compression   (compression.cu / NPP + custom kernels)
//     GPU block-DCT compress each board image (JPEG-style).
//     Save reconstructed image to results/compressed/.
//     Record PSNR and non-zero coefficient ratio in results/compression.csv.
//
//   Stage 3 — Board Evaluation  (evaluator.cu / cuBLAS + cuFFT + Thrust)
//     Evaluate each board from its piece-state (not image pixels).
//     Piece-square tables in constant memory, cuBLAS spatial features,
//     cuFFT pawn structure, Thrust ranking.
//     Results in results/evaluation.csv.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "board_gen.h"
#include "compression.h"
#include "evaluator.h"
#include "image_io.h"

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n\n"
        "Options:\n"
        "  --boards  <N>   Number of boards to generate      (default: 20)\n"
        "  --quality <Q>   DCT compression quality 1-100     (default: 50)\n"
        "  --output  <dir> Output root directory             (default: results)\n"
        "  --seed    <S>   cuRAND seed                       (default: 42)\n"
        "  --top     <N>   Boards to show in summary table   (default: 5)\n"
        "  --verbose       Print per-board timing\n"
        "  --help\n",
        prog);
}

static void mkdirp(const char *path)
{
#ifdef _WIN32
    mkdir(path);
#else
    mkdir(path, 0755);
#endif
}

static void print_gpu_info(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        printf("  GPU  : [not available in this environment]\n\n");
        return;
    }
    int dev = 0; cudaDeviceProp p;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&p, dev);
    printf("  GPU  : %s\n", p.name);
    printf("  SMs  : %d  |  VRAM : %.1f GiB  |  Compute : %d.%d\n\n",
           p.multiProcessorCount,
           (double)p.totalGlobalMem / (1 << 30),
           p.major, p.minor);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    int   n_boards  = 20;
    int   quality   = 50;
    int   top_n     = 5;
    int   verbose   = 0;
    unsigned long long seed = 42ULL;
    const char *out_root = "results";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i],"--boards")  && i+1<argc) n_boards  = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--quality") && i+1<argc) quality   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--output")  && i+1<argc) out_root  = argv[++i];
        else if (!strcmp(argv[i],"--seed")    && i+1<argc) seed      = (unsigned long long)atoll(argv[++i]);
        else if (!strcmp(argv[i],"--top")     && i+1<argc) top_n     = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--verbose"))              verbose   = 1;
        else if (!strcmp(argv[i],"--help"))   { usage(argv[0]); return 0; }
        else { fprintf(stderr,"Unknown arg: %s\n",argv[i]); usage(argv[0]); return 1; }
    }

    if (quality < 1)   quality = 1;
    if (quality > 100) quality = 100;
    if (top_n > n_boards) top_n = n_boards;

    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║        CUDA Chess Vision  (Redesigned)           ║\n");
    printf("║  Stage1: cuRAND  Stage2: DCT  Stage3: cuBLAS     ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");
    print_gpu_info();

    // Create output directories
    char dir_boards[1024], dir_comp[1024];
    snprintf(dir_boards, sizeof(dir_boards), "%s/boards",     out_root);
    snprintf(dir_comp,   sizeof(dir_comp),   "%s/compressed", out_root);
    mkdirp(out_root);
    mkdirp(dir_boards);
    mkdirp(dir_comp);

    // Arrays for all boards
    float      *all_states    = (float *)     malloc(n_boards * 64 * sizeof(float));
    const char **all_filenames= (const char **)malloc(n_boards * sizeof(char *));
    char       **fname_buf    = (char **)      malloc(n_boards * sizeof(char *));
    for (int i = 0; i < n_boards; i++) {
        fname_buf[i] = (char *)malloc(1024);
        snprintf(fname_buf[i], 1024, "%s/board_%03d.png", dir_boards, i);
        all_filenames[i] = fname_buf[i];
    }
    BoardEval  *all_evals = (BoardEval *)malloc(n_boards * sizeof(BoardEval));

    // -------------------------------------------------------------------------
    // Stage 1 — Board Generation (cuRAND)
    // -------------------------------------------------------------------------
    printf("══════════════════════════════════════════════════════\n");
    printf(" Stage 1 │ Board Generation (cuRAND)\n");
    printf("══════════════════════════════════════════════════════\n");
    printf("  Generating %d random legal positions on GPU (seed=%llu)…\n",
           n_boards, seed);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    ChessBoard *boards = boards_generate_gpu(n_boards, seed);
    if (!boards) {
        fprintf(stderr, "  [ERROR] Board generation failed\n"); return 1;
    }

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms1 = 0; cudaEventElapsedTime(&ms1, t0, t1);

    // Render PNGs + convert to float arrays
    int rendered = 0;
    for (int i = 0; i < n_boards; i++) {
        board_to_floats(&boards[i], all_states + i * 64);
        if (board_save_png(fname_buf[i], &boards[i], 512)) {
            rendered++;
            if (verbose) printf("  [%3d/%d] %s\n", i+1, n_boards, fname_buf[i]);
        } else {
            fprintf(stderr, "  [WARN] Failed to render board %d\n", i);
        }
    }
    free(boards);

    printf("  Rendered  : %d boards → %s/\n", rendered, dir_boards);
    printf("  GPU time  : %.2f ms  (%.0f boards/s)\n\n",
           ms1, n_boards / (ms1 / 1000.f));

    // -------------------------------------------------------------------------
    // Stage 2 — DCT Compression (NPP + custom kernels)
    // -------------------------------------------------------------------------
    printf("══════════════════════════════════════════════════════\n");
    printf(" Stage 2 │ DCT Compression (NPP + Block-DCT kernels)\n");
    printf("══════════════════════════════════════════════════════\n");
    printf("  Quality : Q=%d  |  Compressing %d boards…\n", quality, rendered);

    char comp_csv[1024];
    snprintf(comp_csv, sizeof(comp_csv), "%s/compression.csv", out_root);

    float total_psnr = 0.f; int comp_count = 0;
    cudaEventRecord(t0);

    for (int i = 0; i < n_boards; i++) {
        Image img = {0};
        if (!image_load_png(fname_buf[i], &img)) continue;

        CompressedImage c = {0};
        if (!compression_run(&img, quality, &c)) {
            image_free(&img); continue;
        }
        image_free(&img);

        char cpath[1024];
        snprintf(cpath, sizeof(cpath), "%s/board_%03d.png", dir_comp, i);
        compression_save_png(cpath, &c);

        const char *bn = strrchr(fname_buf[i], '/');
        bn = bn ? bn+1 : fname_buf[i];
        compression_csv_append(comp_csv, bn, quality, c.psnr, c.ratio);

        total_psnr += c.psnr;
        comp_count++;

        if (verbose)
            printf("  [%3d/%d] PSNR=%.1f dB  coeff_density=%.1f%%  → %s\n",
                   i+1, n_boards, c.psnr, c.ratio*100.f, cpath);

        compression_free(&c);
    }

    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms2 = 0; cudaEventElapsedTime(&ms2, t0, t1);

    if (comp_count > 0) {
        printf("  Avg PSNR  : %.2f dB\n", total_psnr / comp_count);
        printf("  Saved     : %s\n", comp_csv);
    }
    printf("  GPU time  : %.2f ms\n\n", ms2);

    // -------------------------------------------------------------------------
    // Stage 3 — Board Evaluation (cuBLAS + cuFFT + Thrust)
    // -------------------------------------------------------------------------
    printf("══════════════════════════════════════════════════════\n");
    printf(" Stage 3 │ Board Evaluation (cuBLAS + cuFFT + Thrust)\n");
    printf("══════════════════════════════════════════════════════\n");
    printf("  Evaluating %d boards…\n", rendered);

    cudaEventRecord(t0);
    int eval_ok = evaluator_run(all_states, all_filenames, n_boards, all_evals);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms3 = 0; cudaEventElapsedTime(&ms3, t0, t1);

    if (eval_ok) {
        printf("  GPU time  : %.2f ms\n\n", ms3);

        // Top-N table
        printf("  ┌─────┬──────────────────────┬────────┬──────────┬──────────┬──────────┐\n");
        printf("  │Rank │ File                 │ Score  │ Material │ Position │ PawnFFT  │\n");
        printf("  ├─────┼──────────────────────┼────────┼──────────┼──────────┼──────────┤\n");

        for (int rank = 0; rank < top_n; rank++) {
            for (int i = 0; i < n_boards; i++) {
                if (all_evals[i].rank == rank) {
                    const char *bn = strrchr(all_evals[i].filename, '/');
                    bn = bn ? bn+1 : all_evals[i].filename;
                    printf("  │ #%-3d│ %-20.20s │%7.3f │%9.3f │%9.3f │%9.3f │\n",
                           rank+1, bn,
                           all_evals[i].score,
                           all_evals[i].material,
                           all_evals[i].positional,
                           all_evals[i].pawn_structure);
                }
            }
        }
        printf("  └─────┴──────────────────────┴────────┴──────────┴──────────┴──────────┘\n\n");

        char eval_csv[1024];
        snprintf(eval_csv, sizeof(eval_csv), "%s/evaluation.csv", out_root);
        evaluator_save_csv(eval_csv, all_evals, n_boards);
        printf("  Saved     : %s\n\n", eval_csv);
    } else {
        fprintf(stderr, "  [ERROR] Evaluation failed\n\n");
    }

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    printf("══════════════════════════════════════════════════════\n");
    printf(" Summary\n");
    printf("══════════════════════════════════════════════════════\n");
    printf("  Boards generated  : %d\n", n_boards);
    printf("  Boards rendered   : %d  →  %s/\n", rendered, dir_boards);
    printf("  Boards compressed : %d  →  %s/\n", comp_count, dir_comp);
    printf("  Stage 1 (cuRAND)  : %.2f ms\n", ms1);
    printf("  Stage 2 (DCT)     : %.2f ms\n", ms2);
    printf("  Stage 3 (eval)    : %.2f ms\n", ms3);
    printf("  Total GPU time    : %.2f ms\n", ms1 + ms2 + ms3);
    printf("\n  Visualise: python3 scripts/visualize.py --results %s\n\n",
           out_root);

    // Cleanup
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    free(all_states); free(all_evals); free(all_filenames);
    for (int i = 0; i < n_boards; i++) free(fname_buf[i]);
    free(fname_buf);
    return 0;
}