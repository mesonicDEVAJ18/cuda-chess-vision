// compression.cu — Stage 2: GPU block-DCT image compression (JPEG-style)
//
// Pipeline per image:
//   NPP  : nppiRGBToGray_8u_C3C1R         RGB → grayscale
//   k_dct8x8_forward                       per-block 2-D DCT (separable, shared mem)
//   k_quantize                             JPEG luma Q-table + quality scaling
//   k_idct8x8                              dequantize + inverse DCT → uint8
//   k_psnr_reduce                          parallel reduction → sum-of-sq-diffs
//
// The DCT works on 8×8 blocks; image dimensions must be multiples of 8.
// (All boards are rendered at 512×512, so this is always satisfied.)

#include <cuda_runtime.h>
#include <nppi.h>
#include <npp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <png.h>

#include "compression.h"
#include "image_io.h"

#define NPP_CHK(x) do { \
    NppStatus _s=(x); \
    if(_s!=NPP_SUCCESS){fprintf(stderr,"[compr] NPP %d @ %s:%d\n",(int)_s,__FILE__,__LINE__);return 0;} \
} while(0)

#define CUDA_CHK(x) do { \
    cudaError_t _e=(x); \
    if(_e!=cudaSuccess){fprintf(stderr,"[compr] CUDA %s @ %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);return 0;} \
} while(0)

// ---------------------------------------------------------------------------
// JPEG luminance quantisation table (zigzag → row-major 8×8)
// ---------------------------------------------------------------------------
__constant__ int16_t d_qtable[64] = {
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68,109,103, 77,
    24, 35, 55, 64, 81,104,113, 92,
    49, 64, 78, 87,103,121,120,101,
    72, 92, 95, 98,112,100,103, 99
};

// Pre-computed DCT cosine table: cos_tab[u][x] = C(u)*cos((2x+1)*u*PI/16)
// Stored in constant memory to avoid recomputation each kernel call.
__constant__ float d_cos[8][8];

// Host-side cosine table initialisation (called once before first kernel)
static int cos_table_uploaded = 0;
static void upload_cos_table(void)
{
    if (cos_table_uploaded) return;
    float h_cos[8][8];
    const float PI = 3.14159265358979323846f;
    for (int u = 0; u < 8; u++) {
        float cu = (u == 0) ? 1.f / sqrtf(2.f) : 1.f;
        for (int x = 0; x < 8; x++)
            h_cos[u][x] = cu * cosf((2*x + 1) * u * PI / 16.f);
    }
    cudaMemcpyToSymbol(d_cos, h_cos, sizeof(h_cos));
    cos_table_uploaded = 1;
}

// ---------------------------------------------------------------------------
// Kernel 1 — forward 2-D DCT on 8×8 blocks
// Grid : (W/8, H/8) blocks    Block : (8, 8) threads
// Thread (tx,ty) computes F[ty][tx] for its image block.
// Level-shift: subtract 128 before DCT.
// ---------------------------------------------------------------------------
__global__ void k_dct8x8_forward(
    const uint8_t * __restrict__ src,   // [H × W] grayscale
    float         * __restrict__ dct,   // [H × W] float coefficients
    int W, int H)
{
    __shared__ float s[8][8]; // pixel tile
    __shared__ float t[8][8]; // intermediate (after row DCT)

    int bx = blockIdx.x * 8, by = blockIdx.y * 8;
    int tx = threadIdx.x,    ty = threadIdx.y;

    // Load pixel, level-shift
    int px = bx + tx, py = by + ty;
    s[ty][tx] = (px < W && py < H) ? (float)src[py*W + px] - 128.f : 0.f;
    __syncthreads();

    // Row DCT: thread (tx,ty) computes coefficient for frequency (tx) along row ty
    {
        float sum = 0.f;
        for (int x = 0; x < 8; x++) sum += s[ty][x] * d_cos[tx][x];
        t[ty][tx] = sum * 0.25f;      // 2/N * 0.5 = 0.25 (absorbed into formula)
    }
    __syncthreads();

    // Column DCT: thread (tx,ty) computes final F[ty][tx]
    {
        float sum = 0.f;
        for (int y = 0; y < 8; y++) sum += t[y][tx] * d_cos[ty][y];
        float val = sum * 0.25f;
        if (px < W && py < H) dct[py*W + px] = val;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2 — quantise DCT coefficients, count non-zero
// Grid : (W/8, H/8)    Block : (8, 8)
// ---------------------------------------------------------------------------
__global__ void k_quantize(
    const float    * __restrict__ dct,    // [H × W] input floats
    int16_t        * __restrict__ quant,  // [H × W] quantised output
    unsigned int   * __restrict__ nonzero,// single counter
    int quality, int W, int H)
{
    int bx = blockIdx.x * 8, by = blockIdx.y * 8;
    int tx = threadIdx.x,    ty = threadIdx.y;
    int px = bx + tx,        py = by + ty;
    if (px >= W || py >= H) return;

    int uv = ty * 8 + tx;   // coefficient index within block
    int qt = (int)d_qtable[uv];

    // JPEG quality scaling
    int scale = (quality < 50) ? (5000 / quality) : (200 - 2 * quality);
    int sq = (qt * scale + 50) / 100;
    if (sq < 1) sq = 1;
    if (sq > 255) sq = 255;

    float coeff = dct[py*W + px];
    int16_t q = (int16_t)floorf(coeff / sq + 0.5f);
    quant[py*W + px] = q;

    if (q != 0) atomicAdd(nonzero, 1u);
}

// ---------------------------------------------------------------------------
// Kernel 3 — dequantise + inverse 2-D DCT → uint8 output
// Grid : (W/8, H/8)    Block : (8, 8)
// ---------------------------------------------------------------------------
__global__ void k_idct8x8(
    const int16_t  * __restrict__ quant, // [H × W]
    uint8_t        * __restrict__ out,   // [H × W]
    int quality, int W, int H)
{
    __shared__ float s[8][8]; // dequantised coefficients
    __shared__ float t[8][8]; // after column IDCT

    int bx = blockIdx.x * 8, by = blockIdx.y * 8;
    int tx = threadIdx.x,    ty = threadIdx.y;
    int px = bx + tx,        py = by + ty;

    // Dequantise
    int uv = ty * 8 + tx;
    int qt = (int)d_qtable[uv];
    int scale = (quality < 50) ? (5000 / quality) : (200 - 2 * quality);
    int sq = (qt * scale + 50) / 100;
    if (sq < 1) sq = 1; if (sq > 255) sq = 255;

    s[ty][tx] = (px < W && py < H) ? (float)quant[py*W + px] * sq : 0.f;
    __syncthreads();

    // Column IDCT: reconstruct along column axis
    // Thread (tx,ty) accumulates over frequency index ty
    {
        float sum = 0.f;
        for (int v = 0; v < 8; v++) sum += s[v][tx] * d_cos[v][ty];
        t[ty][tx] = sum * 0.25f;
    }
    __syncthreads();

    // Row IDCT: reconstruct along row axis
    {
        float sum = 0.f;
        for (int u = 0; u < 8; u++) sum += t[ty][u] * d_cos[u][tx];
        float pixel = sum * 0.25f + 128.f;   // reverse level-shift
        int clamped = (int)(pixel + 0.5f);
        if (clamped < 0)   clamped = 0;
        if (clamped > 255) clamped = 255;
        if (px < W && py < H) out[py*W + px] = (uint8_t)clamped;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4 — parallel PSNR reduction (sum of squared differences)
// Each thread handles one pixel; shared memory reduction per block.
// ---------------------------------------------------------------------------
__global__ void k_psnr_reduce(
    const uint8_t * __restrict__ orig,
    const uint8_t * __restrict__ rec,
    float         * __restrict__ block_sums,
    int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float diff = 0.f;
    if (idx < N) {
        float d = (float)orig[idx] - (float)rec[idx];
        diff = d * d;
    }
    sdata[tid] = diff;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) block_sums[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// compression_run — host entry point
// ---------------------------------------------------------------------------
int compression_run(const Image *img, int quality, CompressedImage *out)
{
    int W = img->width, H = img->height, N = W * H;
    if (W % 8 != 0 || H % 8 != 0) {
        fprintf(stderr, "[compr] Image dimensions must be multiples of 8 (%dx%d)\n", W, H);
        return 0;
    }
    if (quality < 1)   quality = 1;
    if (quality > 100) quality = 100;

    upload_cos_table();

    // Device buffers
    uint8_t      *d_rgb=NULL, *d_gray=NULL, *d_out=NULL;
    float        *d_dct=NULL;
    int16_t      *d_quant=NULL;
    unsigned int *d_nonzero=NULL;
    float        *d_block_sums=NULL;

    int ret = 0;
    int threads_psnr = 256;
    int blocks_psnr  = (N + threads_psnr - 1) / threads_psnr;

    CUDA_CHK(cudaMalloc(&d_rgb,   N*3));
    CUDA_CHK(cudaMalloc(&d_gray,  N));
    CUDA_CHK(cudaMalloc(&d_dct,   N*sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_quant, N*sizeof(int16_t)));
    CUDA_CHK(cudaMalloc(&d_out,   N));
    CUDA_CHK(cudaMalloc(&d_nonzero, sizeof(unsigned int)));
    CUDA_CHK(cudaMalloc(&d_block_sums, blocks_psnr*sizeof(float)));

    CUDA_CHK(cudaMemcpy(d_rgb, img->data, N*3, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemset(d_nonzero, 0, sizeof(unsigned int)));

    // NPP: RGB → grayscale
    NppiSize roi = {W, H};
    NPP_CHK(nppiRGBToGray_8u_C3C1R(d_rgb, W*3, d_gray, W, roi));

    // Forward DCT
    dim3 blk(8,8), grd(W/8, H/8);
    k_dct8x8_forward<<<grd,blk>>>(d_gray, d_dct, W, H);
    CUDA_CHK(cudaGetLastError());

    // Quantise
    k_quantize<<<grd,blk>>>(d_dct, d_quant, d_nonzero, quality, W, H);
    CUDA_CHK(cudaGetLastError());

    // Inverse DCT
    k_idct8x8<<<grd,blk>>>(d_quant, d_out, quality, W, H);
    CUDA_CHK(cudaGetLastError());

    // PSNR reduction
    k_psnr_reduce<<<blocks_psnr, threads_psnr, threads_psnr*sizeof(float)>>>(
        d_gray, d_out, d_block_sums, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // Download PSNR partial sums
    {
        float *h_sums = (float *)malloc(blocks_psnr * sizeof(float));
        unsigned int h_nonzero = 0;
        CUDA_CHK(cudaMemcpy(h_sums, d_block_sums,
                            blocks_psnr*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(&h_nonzero, d_nonzero,
                            sizeof(unsigned int), cudaMemcpyDeviceToHost));

        double sum = 0.0;
        for (int i = 0; i < blocks_psnr; i++) sum += h_sums[i];
        free(h_sums);

        double mse  = sum / (double)N;
        float  psnr = (mse > 0.0) ? (float)(10.0 * log10(255.0*255.0 / mse)) : 99.0f;

        out->psnr    = psnr;
        out->ratio   = (float)h_nonzero / (float)N;
        out->quality = quality;
        out->width   = W;
        out->height  = H;
    }

    // Download reconstructed grayscale image
    out->gray = (uint8_t *)malloc(N);
    if (!out->gray) goto done;
    CUDA_CHK(cudaMemcpy(out->gray, d_out, N, cudaMemcpyDeviceToHost));
    ret = 1;

done:
    cudaFree(d_rgb); cudaFree(d_gray); cudaFree(d_dct);
    cudaFree(d_quant); cudaFree(d_out);
    cudaFree(d_nonzero); cudaFree(d_block_sums);
    return ret;
}

// ---------------------------------------------------------------------------
// compression_save_png
// ---------------------------------------------------------------------------
int compression_save_png(const char *path, const CompressedImage *c)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "[compr] Cannot write: %s\n", path); return 0; }

    png_structp png  = png_create_write_struct(PNG_LIBPNG_VER_STRING,NULL,NULL,NULL);
    png_infop   info = png_create_info_struct(png);
    if (!png || !info) { fclose(fp); return 0; }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info); fclose(fp); return 0;
    }
    png_init_io(png, fp);
    png_set_IHDR(png, info, c->width, c->height, 8,
                 PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    for (int y = 0; y < c->height; y++)
        png_write_row(png, c->gray + y * c->width);
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return 1;
}

// ---------------------------------------------------------------------------
// compression_free
// ---------------------------------------------------------------------------
void compression_free(CompressedImage *c)
{
    if (c && c->gray) { free(c->gray); c->gray = NULL; }
}
