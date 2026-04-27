// pipeline.cu — GPU image processing pipeline
// NPP: RGBToGray, GaussBlur, SobelX, SobelY
// Custom kernels: edge combination, histogram equalization, per-square intensities

#include <cuda_runtime.h>
#include <nppi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pipeline.h"
#include "image_io.h"
#include <npp.h>

#define NPP_CHECK(x) do { \
    NppStatus st = (x); \
    if (st != NPP_SUCCESS) { \
        printf("[NPP ERROR] code=%d at %s:%d\n", st, __FILE__, __LINE__); \
        return 0; \
    } \
} while(0)

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA ERROR: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        return 0; \
    } \
} while(0)

#define CUDA_SYNC_CHECK() do { \
    cudaError_t err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        printf("CUDA SYNC ERROR: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        return 0; \
    } \
} while(0)

// Kernel 1: gradient magnitude sqrt(Gx^2 + Gy^2)
__global__ void k_combine_edges(
    const uint8_t * __restrict__ gx,
    const uint8_t * __restrict__ gy,
    uint8_t       * __restrict__ mag,
    int W, int H)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x>=W || y>=H) return;
    int i = y*W+x;
    float m = sqrtf((float)gx[i]*gx[i] + (float)gy[i]*gy[i]);
    mag[i] = (uint8_t)(m > 255.f ? 255 : (int)m);
}

// Kernel 2: 256-bin histogram with shared memory atomics
__global__ void k_histogram(
    const uint8_t * __restrict__ img,
    unsigned int  * __restrict__ hist,
    int N)
{
    __shared__ unsigned int lh[256];
    lh[threadIdx.x] = 0; __syncthreads();
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = gridDim.x*blockDim.x;
    while (i < N) { atomicAdd(&lh[img[i]], 1); i += stride; }
    __syncthreads();
    atomicAdd(&hist[threadIdx.x], lh[threadIdx.x]);
}

// Kernel 3: apply LUT for histogram equalization
__global__ void k_apply_lut(
    const uint8_t * __restrict__ src,
    uint8_t       * __restrict__ dst,
    const uint8_t * __restrict__ lut,
    int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) dst[i] = lut[src[i]];
}

// Kernel 4: per-square mean intensity (8x8 grid, shared-mem reduction)
__global__ void k_square_intensities(
    const uint8_t * __restrict__ img,
    float         * __restrict__ intensities,
    int W, int H)
{
    int sq_col = blockIdx.x, sq_row = blockIdx.y;
    int sq_w = W/8, sq_h = H/8;
    int x0 = sq_col*sq_w, y0 = sq_row*sq_h;
    int tid = threadIdx.y*blockDim.x + threadIdx.x;
    int nt  = blockDim.x*blockDim.y;

    __shared__ float s[256];
    s[tid] = 0.f; __syncthreads();

    int total = sq_w*sq_h;
    for (int k=tid; k<total; k+=nt) {
        int dx=k%sq_w, dy=k/sq_w;
        s[tid] += (float)img[(y0+dy)*W + (x0+dx)];
    }
    __syncthreads();
    for (int st=128; st>0; st>>=1) {
        if (tid<st) s[tid]+=s[tid+st]; __syncthreads();
    }
    if (tid==0) intensities[sq_row*8+sq_col] = s[0]/(float)total;
}

static void build_equalize_lut(const unsigned int *hist, int N, uint8_t *lut) {
    unsigned long long cdf=0, cdf_min=0; int found=0;
    for (int i=0; i<256; i++) {
        cdf += hist[i];
        if (hist[i]>0 && !found) { cdf_min=cdf; found=1; }
        int denom = N-(int)cdf_min;
        lut[i] = denom>0 ? (uint8_t)((double)(cdf-cdf_min)/denom*255.0+0.5) : (uint8_t)i;
    }
}

static int npp_ok(NppStatus s, const char *stage) {
    if (s!=NPP_SUCCESS) { fprintf(stderr,"[pipeline] NPP '%s': %d\n",stage,(int)s); return 0; }
    return 1;
}

int pipeline_run(const Image *img, PipelineResult *result) {
    int W=img->width, H=img->height, N=W*H;
    NppiSize roi={W,H};
    int step3=W*3, step1=W;

    uint8_t      *d_rgb=NULL,*d_gray=NULL,*d_blur=NULL;
    uint8_t      *d_sx=NULL,*d_sy=NULL,*d_mag=NULL,*d_eq=NULL,*d_lut=NULL;
    unsigned int *d_hist=NULL;
    float        *d_intens=NULL;

    cudaMalloc(&d_rgb,   N*3);  cudaMalloc(&d_gray,  N);
    cudaMalloc(&d_blur,  N);    cudaMalloc(&d_sx,    N);
    cudaMalloc(&d_sy,    N);    cudaMalloc(&d_mag,   N);
    cudaMalloc(&d_eq,    N);    cudaMalloc(&d_lut,   256);
    cudaMalloc(&d_hist,  256*sizeof(unsigned int));
    cudaMalloc(&d_intens,64*sizeof(float));

    if (!d_rgb||!d_gray||!d_blur||!d_sx||!d_sy||!d_mag||!d_eq||!d_lut||!d_hist||!d_intens)
        goto fail;

    cudaMemcpy(d_rgb, img->data, N*3, cudaMemcpyHostToDevice);

    if (!npp_ok(nppiRGBToGray_8u_C3C1R  (d_rgb,step3,d_gray,step1,roi),"RGBToGray")) goto fail;
    if (!npp_ok(nppiFilterGauss_8u_C1R  (d_gray,step1,d_blur,step1,roi,NPP_MASK_SIZE_3_X_3),"Gauss")) goto fail;
    if (!npp_ok(nppiFilterSobelHoriz_8u_C1R(d_blur,step1,d_sx,step1,roi),"SobelX")) goto fail;
    if (!npp_ok(nppiFilterSobelVert_8u_C1R (d_blur,step1,d_sy,step1,roi),"SobelY")) goto fail;

    { dim3 bl(16,16), gr((W+15)/16,(H+15)/16);
      k_combine_edges<<<gr,bl>>>(d_sx,d_sy,d_mag,W,H); }

    { cudaMemset(d_hist,0,256*sizeof(unsigned int));
      int gs=min((N+255)/256,65535);
      k_histogram<<<gs,256>>>(d_mag,d_hist,N);
      cudaDeviceSynchronize();
      unsigned int h_hist[256];
      cudaMemcpy(h_hist,d_hist,256*sizeof(unsigned int),cudaMemcpyDeviceToHost);
      uint8_t lut[256]; build_equalize_lut(h_hist,N,lut);
      cudaMemcpy(d_lut,lut,256,cudaMemcpyHostToDevice);
      k_apply_lut<<<(N+255)/256,256>>>(d_mag,d_eq,d_lut,N);
      cudaDeviceSynchronize(); }

    { dim3 gr(8,8),bl(16,16);
      k_square_intensities<<<gr,bl>>>(d_eq,d_intens,W,H);
      cudaDeviceSynchronize(); }

    result->gray   = (uint8_t*)malloc(N);
    result->edges  = (uint8_t*)malloc(N);
    result->width  = W; result->height = H;
    cudaMemcpy(result->gray,       d_eq,    N,              cudaMemcpyDeviceToHost);
    cudaMemcpy(result->edges,      d_mag,   N,              cudaMemcpyDeviceToHost);
    cudaMemcpy(result->intensities,d_intens,64*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_rgb);cudaFree(d_gray);cudaFree(d_blur);cudaFree(d_sx);cudaFree(d_sy);
    cudaFree(d_mag);cudaFree(d_eq);cudaFree(d_lut);cudaFree(d_hist);cudaFree(d_intens);
    return 1;
fail:
    cudaFree(d_rgb);cudaFree(d_gray);cudaFree(d_blur);cudaFree(d_sx);cudaFree(d_sy);
    cudaFree(d_mag);cudaFree(d_eq);cudaFree(d_lut);cudaFree(d_hist);cudaFree(d_intens);
    return 0;
}

void pipeline_result_free(PipelineResult *r) {
    if (r->gray)  { free(r->gray);  r->gray=NULL;  }
    if (r->edges) { free(r->edges); r->edges=NULL; }
}
