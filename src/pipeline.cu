// pipeline.cu — GPU image processing pipeline
// Stages:
//   1. RGB → Grayscale        (NPP: nppiRGBToGray_8u_C3C1R)
//   2. Gaussian Blur 3×3      (NPP: nppiFilterGauss_8u_C1R)
//   3. Sobel X                (NPP: nppiFilterSobelHoriz_8u_C1R)
//   4. Sobel Y                (NPP: nppiFilterSobelVert_8u_C1R)
//   5. Edge magnitude         (custom kernel: sqrt(Gx² + Gy²))
//   6. Histogram equalization (NPP: nppiEqualizeHist_8u_C1R)
//   7. Per-square intensities (custom kernel: 8×8 shared-mem reduction)

#include <cuda_runtime.h>
#include <nppi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "pipeline.h"
#include "image_io.h"

// -----------------------------------------------------------------------
// Kernel 1: combine Sobel X and Y into gradient magnitude
//   magnitude[i] = clamp(sqrt(gx[i]^2 + gy[i]^2), 0, 255)
// -----------------------------------------------------------------------
__global__ void k_combine_edges(
    const uint8_t * __restrict__ gx,
    const uint8_t * __restrict__ gy,
    uint8_t       * __restrict__ mag,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int i = y * width + x;
    float m = sqrtf((float)gx[i] * gx[i] + (float)gy[i] * gy[i]);
    mag[i] = (uint8_t)(m > 255.0f ? 255 : m);
}

// -----------------------------------------------------------------------
// Kernel 2: extract per-square mean intensity
//   Divides the image into an 8×8 grid of chess squares.
//   One block per square (gridDim = {8,8}).
//   Threads within a block cooperatively sum pixel values via shared memory.
// -----------------------------------------------------------------------
__global__ void k_square_intensities(
    const uint8_t * __restrict__ img,
    float         * __restrict__ intensities,
    int width, int height)
{
    // Which chess square
    int sq_col = blockIdx.x;   // file 0–7 (a–h)
    int sq_row = blockIdx.y;   // rank 0–7

    int sq_w = width  / 8;
    int sq_h = height / 8;
    int x0   = sq_col * sq_w;
    int y0   = sq_row * sq_h;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;

    __shared__ float sdata[256];
    sdata[tid] = 0.0f;
    __syncthreads();

    // Each thread sums a strided subset of pixels in this square
    int total = sq_w * sq_h;
    for (int k = tid; k < total; k += nthreads) {
        int dx = k % sq_w;
        int dy = k / sq_w;
        sdata[tid] += (float)img[(y0 + dy) * width + (x0 + dx)];
    }
    __syncthreads();

    // Parallel reduction
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        intensities[sq_row * 8 + sq_col] = sdata[0] / (float)total;
}

// -----------------------------------------------------------------------
// Helper: check NPP status and print message
// -----------------------------------------------------------------------
static int npp_ok(NppStatus s, const char *stage) {
    if (s != NPP_SUCCESS) {
        fprintf(stderr, "[pipeline] NPP error at '%s': %d\n", stage, (int)s);
        return 0;
    }
    return 1;
}

// -----------------------------------------------------------------------
// pipeline_run: the full GPU pipeline for one image
// -----------------------------------------------------------------------
int pipeline_run(const Image *img, PipelineResult *result) {
    int W = img->width;
    int H = img->height;
    int N = W * H;

    NppiSize roi = { W, H };
    int step3 = W * 3;   // bytes per row, 3-channel
    int step1 = W;       // bytes per row, 1-channel

    // --- Allocate device buffers ---
    uint8_t *d_rgb = NULL, *d_gray = NULL, *d_blur = NULL;
    uint8_t *d_sx  = NULL, *d_sy   = NULL, *d_mag  = NULL, *d_eq = NULL;
    float   *d_intens = NULL;

    cudaMalloc(&d_rgb,    N * 3);
    cudaMalloc(&d_gray,   N);
    cudaMalloc(&d_blur,   N);
    cudaMalloc(&d_sx,     N);
    cudaMalloc(&d_sy,     N);
    cudaMalloc(&d_mag,    N);
    cudaMalloc(&d_eq,     N);
    cudaMalloc(&d_intens, 64 * sizeof(float));

    // Check all allocations
    if (!d_rgb || !d_gray || !d_blur || !d_sx || !d_sy || !d_mag || !d_eq || !d_intens) {
        fprintf(stderr, "[pipeline] cudaMalloc failed\n");
        goto fail;
    }

    // --- Upload RGB ---
    cudaMemcpy(d_rgb, img->data, N * 3, cudaMemcpyHostToDevice);

    // --- Stage 1: RGB → Grayscale ---
    if (!npp_ok(nppiRGBToGray_8u_C3C1R(d_rgb, step3, d_gray, step1, roi), "RGBToGray"))
        goto fail;

    // --- Stage 2: Gaussian Blur 3×3 ---
    if (!npp_ok(nppiFilterGauss_8u_C1R(d_gray, step1, d_blur, step1, roi, NPP_MASK_SIZE_3_X_3), "GaussBlur"))
        goto fail;

    // --- Stage 3: Sobel X ---
    if (!npp_ok(nppiFilterSobelHoriz_8u_C1R(d_blur, step1, d_sx, step1, roi), "SobelX"))
        goto fail;

    // --- Stage 4: Sobel Y ---
    if (!npp_ok(nppiFilterSobelVert_8u_C1R(d_blur, step1, d_sy, step1, roi), "SobelY"))
        goto fail;

    // --- Stage 5: Combine edge maps (custom kernel) ---
    {
        dim3 block(16, 16);
        dim3 grid((W + 15) / 16, (H + 15) / 16);
        k_combine_edges<<<grid, block>>>(d_sx, d_sy, d_mag, W, H);
        if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "[pipeline] k_combine_edges failed\n"); goto fail; }
    }

    // --- Stage 6: Histogram equalization on edge map ---
    if (!npp_ok(nppiEqualizeHist_8u_C1R(d_mag, step1, d_eq, step1, roi), "EqualizeHist"))
        goto fail;

    // --- Stage 7: Per-square intensities (on equalized image) ---
    {
        dim3 grid(8, 8);
        dim3 block(16, 16);  // 256 threads; shared mem = 256 * sizeof(float) = 1 KB
        k_square_intensities<<<grid, block>>>(d_eq, d_intens, W, H);
        if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "[pipeline] k_square_intensities failed\n"); goto fail; }
    }

    cudaDeviceSynchronize();

    // --- Download results ---
    result->gray   = (uint8_t *)malloc(N);
    result->edges  = (uint8_t *)malloc(N);
    result->width  = W;
    result->height = H;

    cudaMemcpy(result->gray,        d_eq,     N,                  cudaMemcpyDeviceToHost);
    cudaMemcpy(result->edges,       d_mag,    N,                  cudaMemcpyDeviceToHost);
    cudaMemcpy(result->intensities, d_intens, 64 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rgb); cudaFree(d_gray); cudaFree(d_blur);
    cudaFree(d_sx);  cudaFree(d_sy);   cudaFree(d_mag);
    cudaFree(d_eq);  cudaFree(d_intens);
    return 1;

fail:
    cudaFree(d_rgb); cudaFree(d_gray); cudaFree(d_blur);
    cudaFree(d_sx);  cudaFree(d_sy);   cudaFree(d_mag);
    cudaFree(d_eq);  cudaFree(d_intens);
    return 0;
}

void pipeline_result_free(PipelineResult *r) {
    if (r->gray)  { free(r->gray);  r->gray  = NULL; }
    if (r->edges) { free(r->edges); r->edges = NULL; }
}
