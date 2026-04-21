// main.cu
// CUDA Chess Vision — GPU-accelerated chess board image processing
// CUDA at Scale Independent Project
//
// Processes a directory of chess board PNG images through a GPU pipeline:
//   RGB→Gray | Gaussian Blur | Sobel Edge Detection | Histogram Equalization
// Per-square intensity maps are exported as CSV signals.
//
// Build:  make
// Run:    ./chess_vision --input data/sample_boards --output results --csv --verbose

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "image_io.h"
#include "pipeline.h"

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --input <dir> [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --input   <dir>   PNG image directory (required)\n"
        "  --output  <dir>   Output directory      (default: results)\n"
        "  --batch   <N>     Batch size             (default: 32)\n"
        "  --csv             Export per-square intensity CSVs\n"
        "  --verbose         Print per-image timing\n"
        "  --help\n",
        prog);
}

static void print_gpu_info(void) {
    int dev;
    cudaDeviceProp p;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&p, dev);
    printf("GPU : %s\n", p.name);
    printf("SMs : %d  |  Memory: %.1f GiB  |  Compute: %d.%d\n",
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
        if      (!strcmp(argv[i], "--input")   && i+1 < argc) input_dir  = argv[++i];
        else if (!strcmp(argv[i], "--output")  && i+1 < argc) output_dir = argv[++i];
        else if (!strcmp(argv[i], "--batch")   && i+1 < argc) batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv"))                    export_csv = 1;
        else if (!strcmp(argv[i], "--verbose"))                verbose    = 1;
        else if (!strcmp(argv[i], "--help"))  { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!input_dir) { fprintf(stderr, "Error: --input required\n"); usage(argv[0]); return 1; }

    print_gpu_info();
    mkdir(output_dir, 0755);

    // Collect images
    int    n_images = 0;
    char **paths    = collect_png_files(input_dir, &n_images);
    if (n_images == 0) { fprintf(stderr, "No PNG files in: %s\n", input_dir); return 1; }
    printf("Images : %d  |  Batch size: %d\n\n", n_images, batch_size);

    // Warm up CUDA context
    cudaFree(0);

    // Overall timing
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);
    cudaEventRecord(ev_start);

    int processed = 0;

    for (int b = 0; b < n_images; b += batch_size) {
        int end = b + batch_size < n_images ? b + batch_size : n_images;
        int batch_n = end - b;

        // Load batch into host memory
        Image *host_imgs = (Image *)calloc(batch_n, sizeof(Image));
        int loaded = 0;
        for (int i = 0; i < batch_n; i++) {
            if (image_load_png(paths[b + i], &host_imgs[i]))
                loaded++;
            else
                fprintf(stderr, "  [SKIP] %s\n", paths[b + i]);
        }

        // Process each image in batch through GPU pipeline
        for (int i = 0; i < batch_n; i++) {
            if (!host_imgs[i].data) continue;

            cudaEvent_t t0, t1;
            if (verbose) { cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventRecord(t0); }

            PipelineResult res;
            int ok = pipeline_run(&host_imgs[i], &res);

            if (verbose) {
                cudaEventRecord(t1);
                cudaEventSynchronize(t1);
                float ms = 0;
                cudaEventElapsedTime(&ms, t0, t1);
                const char *bn = strrchr(paths[b+i], '/');
                bn = bn ? bn+1 : paths[b+i];
                printf("  [%3d/%d] %-36s  %.2f ms\n", processed+1, n_images, bn, ms);
                cudaEventDestroy(t0);
                cudaEventDestroy(t1);
            }

            if (!ok) { image_free(&host_imgs[i]); continue; }

            // Save processed image
            const char *bn = strrchr(paths[b+i], '/');
            bn = bn ? bn+1 : paths[b+i];
            char out_img[2048], out_csv[2048];
            snprintf(out_img, sizeof(out_img), "%s/proc_%s", output_dir, bn);
            image_save_png_gray(out_img, res.gray, res.width, res.height);

            if (export_csv) {
                // Strip .png, add .csv
                char base[512];
                strncpy(base, bn, sizeof(base)-1);
                char *dot = strrchr(base, '.');
                if (dot) *dot = '\0';
                snprintf(out_csv, sizeof(out_csv), "%s/intensity_%s.csv", output_dir, base);
                save_intensity_csv(out_csv, res.intensities);
            }

            pipeline_result_free(&res);
            image_free(&host_imgs[i]);
            processed++;
        }
        free(host_imgs);

        int batch_num = b / batch_size + 1;
        int total_batches = (n_images + batch_size - 1) / batch_size;
        printf("Batch %d/%d done  (%d images total)\n", batch_num, total_batches, processed);
    }

    cudaEventRecord(ev_end);
    cudaEventSynchronize(ev_end);
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, ev_start, ev_end);

    printf("\n=== Summary ===\n");
    printf("Processed  : %d / %d images\n", processed, n_images);
    printf("GPU time   : %.2f s\n", total_ms / 1000.0f);
    if (processed > 0)
        printf("Throughput : %.1f img/s\n", processed / (total_ms / 1000.0f));
    printf("Output dir : %s/\n", output_dir);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    for (int i = 0; i < n_images; i++) free(paths[i]);
    free(paths);
    return 0;
}
