// pipeline.cu — GPU image processing pipeline
// Stages:
//   1. RGB → Grayscale        (NPP: nppiRGBToGray_8u_C3C1R)
//   2. Gaussian Blur 3x3      (NPP: nppiFilterGauss_8u_C1R)
//   3. Sobel X                (NPP: nppiFilterSobelHoriz_8u_C1R)
//   4. Sobel Y                (NPP: nppiFilterSobelVert_8u_C1R)
//   5. Edge magnitude         (custom kernel: sqrt(Gx^2 + Gy^2))
//   6. Histogram equalization (custom kernel: avoids deprecated NPP API)
//   7. Per-square intensities (custom kernel: 8x8 shared-mem reduction)

#include <cuda_runtime.h>
#include <nppi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "pipeline.h"
#include "image_io.h"

// ---------------------------------------------------------------------------
// Kernel 1: combine Sobel X and Y into gradient magnitude
// ---------------------------------------------------------------------------
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
    mag[i] = (uint8_t)(m > 255.0f ? 255 : (int)m);
}

// ---------------------------------------------------------------------------
// Kernel 2: build a 256-bin histogram (one block, 256 threads, atomics)
// ---------------------------------------------------------------------------
__global__ void k_histogram(
    const uint8_t * __restrict__ img,
    unsigned int  * __restrict__ hist,
    int n)
{
    __shared__ unsigned int local_hist[256];
    local_hist[threadIdx.x] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    while (i < n) {
        atomicAdd(&local_hist[img[i]], 1);
        i += stride;
    }
    __syncthreads();
    atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
}

// ---------------------------------------------------------------------------
// Kernel 3: apply LUT (lookup table) for histogram equalization
// ---------------------------------------------------------------------------
__global__ void k_apply_lut(
    const uint8_t      * __restrict__ src,
    uint8_t            * __restrict__ dst,
    const unsigned char* __restrict__ lut,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = lut[src[i]];
}

// ---------------------------------------------------------------------------
// CPU helper: build equalization LUT from histogram
// ---------------------------------------------------------------------------
static void build_equalize_lut(const unsigned int *hist, int n_pixels, unsigned char *lut) {
    unsigned long long cdf = 0, cdf_min = 0;
    int found_min = 0;
    for (int i = 0; i < 256; i++) {
        cdf += hist[i];
        if (hist[i] > 0 && !found_min) { cdf_min = cdf; found_min = 1; }
        // lut[i] = round((cdf - cdf_min) / (n_pixels - cdf_min) * 255)
        if (n_pixels - (int)cdf_min > 0)
            lut[i] = (unsigned char)((double)(cdf - cdf_min) / (n_pixels - cdf_min) * 255.0 + 0.5);
        else
            lut[i] = (unsigned char)i;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: per-square mean intensity (8x8 chess grid, shared-mem reduction)
// ---------------------------------------------------------------------------
__global__ void k_square_intensities(
    const uint8_t * __restrict__ img,
    float         * __restrict__ intensities,
    int width, int height)
{
    int sq_col = blockIdx.x;
    int sq_row = blockIdx.y;
    int sq_w = width  / 8;
    int sq_h = height / 8;
    int x0   = sq_col * sq_w;
    int y0   = sq_row * sq_h;

    int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;

    __shared__ float sdata[256];
    sdata[tid] = 0.0f;
    __syncthreads();

    int total = sq_w * sq_h;
    for (int k = tid; k < total; k += nthreads) {
        int dx = k % sq_w;
        int dy = k / sq_w;
        sdata[tid] += (float)img[(y0 + dy) * width + (x0 + dx)];
    }
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        intensities[sq_row * 8 + sq_col] = sdata[0] / (float)total;
}

// ---------------------------------------------------------------------------
// Helper: check NPP status
// ---------------------------------------------------------------------------
static int npp_ok(NppStatus s, const char *stage) {
    if (s != NPP_SUCCESS) {
        fprintf(stderr, "[pipeline] NPP error at '%s': %d\n", stage, (int)s);
        return 0;
    }
    return 1;
}

// ---------------------------------------------------------------------------
// pipeline_run
// ---------------------------------------------------------------------------
int pipeline_run(const Image *img, PipelineResult *result) {
    int W = img->width;
    int H = img->height;
    int N = W * H;

    NppiSize roi = { W, H };
    int step3 = W * 3;
    int step1 = W;

    uint8_t      *d_rgb = NULL, *d_gray = NULL, *d_blur = NULL;
    uint8_t      *d_sx  = NULL, *d_sy   = NULL, *d_mag  = NULL, *d_eq = NULL;
    unsigned int *d_hist = NULL;
    float        *d_intens = NULL;

    cudaMalloc(&d_rgb,    N * 3);
    cudaMalloc(&d_gray,   N);
    cudaMalloc(&d_blur,   N);
    cudaMalloc(&d_sx,     N);
    cudaMalloc(&d_sy,     N);
    cudaMalloc(&d_mag,    N);
    cudaMalloc(&d_eq,     N);
    cudaMalloc(&d_hist,   256 * sizeof(unsigned int));
    cudaMalloc(&d_intens, 64  * sizeof(float));

    if (!d_rgb || !d_gray || !d_blur || !d_sx || !d_sy ||
        !d_mag || !d_eq   || !d_hist || !d_intens) {
        fprintf(stderr, "[pipeline] cudaMalloc failed\n");
        goto fail;
    }

    // Upload
    cudaMemcpy(d_rgb, img->data, N * 3, cudaMemcpyHostToDevice);

    // Stage 1: RGB -> Gray
    if (!npp_ok(nppiRGBToGray_8u_C3C1R(d_rgb, step3, d_gray, step1, roi), "RGBToGray"))
        goto fail;

    // Stage 2: Gaussian blur
    if (!npp_ok(nppiFilterGauss_8u_C1R(d_gray, step1, d_blur, step1, roi, NPP_MASK_SIZE_3_X_3), "GaussBlur"))
        goto fail;

    // Stage 3: Sobel X
    if (!npp_ok(nppiFilterSobelHoriz_8u_C1R(d_blur, step1, d_sx, step1, roi), "SobelX"))
        goto fail;

    // Stage 4: Sobel Y
    if (!npp_ok(nppiFilterSobelVert_8u_C1R(d_blur, step1, d_sy, step1, roi), "SobelY"))
        goto fail;

    // Stage 5: Edge magnitude (custom kernel)
    {
        dim3 block(16, 16);
        dim3 grid((W + 15) / 16, (H + 15) / 16);
        k_combine_edges<<<grid, block>>>(d_sx, d_sy, d_mag, W, H);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "[pipeline] k_combine_edges failed\n");
            goto fail;
        }
    }

    // Stage 6: Histogram equalization (custom — avoids deprecated NPP API)
    {
        // Build histogram on GPU
        cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));
        int bsize = 256;
        int gsize = (N + bsize - 1) / bsize;
        if (gsize > 65535) gsize = 65535;
        k_histogram<<<gsize, bsize>>>(d_mag, d_hist, N);
        cudaDeviceSynchronize();

        // Download histogram, build LUT on CPU
        unsigned int h_hist[256];
        cudaMemcpy(h_hist, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        unsigned char lut[256];
        build_equalize_lut(h_hist, N, lut);

        // Upload LUT and apply
        unsigned char *d_lut = NULL;
        cudaMalloc(&d_lut, 256);
        cudaMemcpy(d_lut, lut, 256, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks  = (N + threads - 1) / threads;
        k_apply_lut<<<blocks, threads>>>(d_mag, d_eq, d_lut, N);
        cudaDeviceSynchronize();
        cudaFree(d_lut);

        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "[pipeline] equalization kernels failed\n");
            goto fail;
        }
    }

    // Stage 7: Per-square intensities
    {
        dim3 grid(8, 8);
        dim3 block(16, 16);
        k_square_intensities<<<grid, block>>>(d_eq, d_intens, W, H);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "[pipeline] k_square_intensities failed\n");
            goto fail;
        }
    }

    // Download results
    result->gray   = (uint8_t *)malloc(N);
    result->edges  = (uint8_t *)malloc(N);
    result->width  = W;
    result->height = H;
    cudaMemcpy(result->gray,        d_eq,     N,                  cudaMemcpyDeviceToHost);
    cudaMemcpy(result->edges,       d_mag,    N,                  cudaMemcpyDeviceToHost);
    cudaMemcpy(result->intensities, d_intens, 64 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rgb);  cudaFree(d_gray); cudaFree(d_blur);
    cudaFree(d_sx);   cudaFree(d_sy);   cudaFree(d_mag);
    cudaFree(d_eq);   cudaFree(d_hist); cudaFree(d_intens);
    return 1;

fail:
    cudaFree(d_rgb);  cudaFree(d_gray); cudaFree(d_blur);
    cudaFree(d_sx);   cudaFree(d_sy);   cudaFree(d_mag);
    cudaFree(d_eq);   cudaFree(d_hist); cudaFree(d_intens);
    return 0;
}

void pipeline_result_free(PipelineResult *r) {
    if (r->gray)  { free(r->gray);  r->gray  = NULL; }
    if (r->edges) { free(r->edges); r->edges = NULL; }
}
