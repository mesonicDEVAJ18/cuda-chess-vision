#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "image_io.h"
#include "pipeline.h"
#include "evaluator.h"

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --input <dir> [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --input   <dir>   PNG image directory        (required)\n"
        "  --output  <dir>   Output directory            (default: results)\n"
        "  --batch   <N>     Images per batch            (default: 32)\n"
        "  --csv             Export CSVs\n"
        "  --verbose         Print per-image timing\n"
        "  --help\n",
        prog);
}

static void print_gpu_info(void) {
    // Warm up MUST happen before device queries
    cudaError_t err = cudaFree(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[WARN] CUDA init: %s\n", cudaGetErrorString(err));
        return;
    }
    int dev; cudaDeviceProp p;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&p, dev);
    printf("GPU : %s\n", p.name);
    printf("SMs : %d  |  Memory: %.1f GiB  |  Compute: %d.%d\n\n",
           p.multiProcessorCount,
           (double)p.totalGlobalMem / (1 << 30),
           p.major, p.minor);
}

int main(int argc, char **argv) {
    const char *input_dir  = NULL;
    const char *output_dir = "results";
    int         batch_size = 32;
    int         export_csv = 0;
    int         verbose    = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--input")  && i+1 < argc) input_dir  = argv[++i];
        else if (!strcmp(argv[i], "--output") && i+1 < argc) output_dir = argv[++i];
        else if (!strcmp(argv[i], "--batch")  && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv"))                   export_csv = 1;
        else if (!strcmp(argv[i], "--verbose"))               verbose    = 1;
        else if (!strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!input_dir) { fprintf(stderr, "Error: --input required\n"); usage(argv[0]); return 1; }

    print_gpu_info();
    mkdir(output_dir, 0755);

    // Collect images
    int    n_images = 0;
    char **paths    = collect_png_files(input_dir, &n_images);
    if (n_images == 0) { fprintf(stderr, "No PNG files in: %s\n", input_dir); return 1; }
    printf("Images : %d  |  Batch: %d\n\n", n_images, batch_size);

    // Print first few paths so we can verify they look correct
    printf("Sample paths:\n");
    for (int i = 0; i < n_images && i < 3; i++)
        printf("  [%d] %s\n", i, paths[i]);
    printf("\n");

    // Allocate storage for all board intensities
    float      *all_intensities = (float *)    malloc(n_images * 64 * sizeof(float));
    const char **all_filenames  = (const char **)malloc(n_images * sizeof(char *));
    BoardEval  *all_evals       = (BoardEval *) malloc(n_images * sizeof(BoardEval));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0);

    int processed = 0;

    // -------------------------------------------------------------------------
    // Stage 1: NPP image pipeline
    // -------------------------------------------------------------------------
    printf("=== Stage 1: Image Pipeline (NPP + custom kernels) ===\n");
    for (int b = 0; b < n_images; b += batch_size) {
        int end = b + batch_size < n_images ? b + batch_size : n_images;

        for (int i = b; i < end; i++) {
            cudaEvent_t t0, t1;
            if (verbose) { cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventRecord(t0); }

            Image img = {0};
            if (!image_load_png(paths[i], &img)) {
                fprintf(stderr, "  [SKIP] Failed to load: %s\n", paths[i]);
                continue;
            }

            PipelineResult res = {0};
            int ok = pipeline_run(&img, &res);
            image_free(&img);
            if (!ok) {
                fprintf(stderr, "  [SKIP] Pipeline failed: %s\n", paths[i]);
                continue;
            }

            // Save processed image
            const char *bn = strrchr(paths[i], '/');
            bn = bn ? bn + 1 : paths[i];
            char out_img[2048];
            snprintf(out_img, sizeof(out_img), "%s/proc_%s", output_dir, bn);
            image_save_png_gray(out_img, res.gray, res.width, res.height);

            if (export_csv) {
                char base[512];
                strncpy(base, bn, 511); base[511] = '\0';
                char *dot = strrchr(base, '.'); if (dot) *dot = '\0';
                char out_csv[2048];
                snprintf(out_csv, sizeof(out_csv), "%s/intensity_%s.csv", output_dir, base);
                save_intensity_csv(out_csv, res.intensities);
            }

            memcpy(all_intensities + processed * 64, res.intensities, 64 * sizeof(float));
            all_filenames[processed] = paths[i];
            pipeline_result_free(&res);

            if (verbose) {
                cudaEventRecord(t1); cudaEventSynchronize(t1);
                float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
                printf("  [%3d/%d] %-36s  %.2f ms\n", processed + 1, n_images, bn, ms);
                cudaEventDestroy(t0); cudaEventDestroy(t1);
            }
            processed++;
        }

        printf("  Batch %d/%d done (%d processed)\n",
               b / batch_size + 1,
               (n_images + batch_size - 1) / batch_size,
               processed);
    }

    // -------------------------------------------------------------------------
    // Stage 2: cuBLAS + cuFFT + Thrust evaluation
    // -------------------------------------------------------------------------
    printf("\n=== Stage 2: Board Evaluation (cuBLAS + cuFFT + Thrust) ===\n");
    if (processed > 0) {
        int ok = evaluator_run(all_intensities, all_filenames, processed, all_evals);
        if (ok) {
            printf("  Evaluated %d boards\n", processed);
            printf("\n  Top 5 boards by GPU evaluation score:\n");
            printf("  %-4s %-30s %8s %10s\n", "Rank", "File", "Score", "PawnFFT");

            for (int rank = 0; rank < 5 && rank < processed; rank++) {
                for (int i = 0; i < processed; i++) {
                    if (all_evals[i].rank == rank) {
                        const char *bn = strrchr(all_evals[i].filename, '/');
                        bn = bn ? bn + 1 : all_evals[i].filename;
                        printf("  #%-3d %-30s %8.4f %10.4f\n",
                               rank + 1, bn,
                               all_evals[i].score,
                               all_evals[i].pawn_structure);
                    }
                }
            }

            if (export_csv) {
                char eval_csv[2048];
                snprintf(eval_csv, sizeof(eval_csv), "%s/evaluation.csv", output_dir);
                evaluator_save_csv(eval_csv, all_evals, processed);
                printf("\n  Saved: %s\n", eval_csv);
            }
        } else {
            fprintf(stderr, "  [ERROR] Evaluation failed\n");
        }
    } else {
        printf("  Skipped (0 images processed in Stage 1)\n");
    }

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float total_ms = 0; cudaEventElapsedTime(&total_ms, ev0, ev1);

    printf("\n=== Summary ===\n");
    printf("Processed  : %d / %d images\n", processed, n_images);
    printf("GPU time   : %.2f s\n", total_ms / 1000.0f);
    if (processed > 0)
        printf("Throughput : %.1f img/s\n", processed / (total_ms / 1000.0f));
    printf("Output dir : %s/\n", output_dir);

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    free(all_intensities); free(all_filenames); free(all_evals);
    for (int i = 0; i < n_images; i++) free(paths[i]);
    free(paths);
    return 0;
}