// evaluator.cu — GPU batch board evaluation
//
// Three GPU libraries in use:
//
// 1. cuBLAS (cublasSgemm) — batch matrix multiply
//    Each board's 64 intensity values form a [1x64] row vector.
//    We stack N boards into a [Nx64] matrix and multiply by a [64x4]
//    weight matrix (piece-square evaluation weights), giving a [Nx4]
//    score matrix. A final dot with a [4x1] aggregation vector yields
//    one scalar score per board.
//
// 2. Thrust — parallel sort + rank assignment
//    After scoring, thrust::sort_by_key sorts all scores descending
//    and writes back the rank of each board among all N boards.
//
// 3. cuFFT (cufftExecR2C) — pawn structure frequency analysis
//    The 8 values of row 1 (rank 2, pawn row for white) are treated
//    as a discrete signal. cuFFT computes the real-to-complex FFT and
//    the energy of high-frequency bins indicates pawn fragmentation.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "evaluator.h"

// ---------------------------------------------------------------------------
// Piece-square weight matrix W [64 x 4], column-major for cuBLAS.
// Each column is one feature:
//   col 0: material proxy   (centre squares weighted higher)
//   col 1: king safety      (corner/edge squares weighted higher)
//   col 2: pawn structure   (rank 1&6 rows weighted)
//   col 3: piece activity   (centre 4x4 weighted strongly)
// Values are in [-1, 1]. Light square = high intensity = white piece present.
// Dark  square = low  intensity = black piece present (negated weight).
// ---------------------------------------------------------------------------
static void build_weight_matrix(float *W) {
    // W is 64 rows x 4 cols, column-major
    for (int sq = 0; sq < 64; sq++) {
        int row = sq / 8;   // rank 0-7
        int col = sq % 8;   // file 0-7

        // Normalise to [-3.5, 3.5] centred
        float dr = row - 3.5f;
        float dc = col - 3.5f;
        float dist_centre = sqrtf(dr*dr + dc*dc);
        float dist_edge   = fminf(fminf(row, 7-row), fminf(col, 7-col));

        // Feature 0: material (centre bonus)
        W[0*64 + sq] = 1.0f - dist_centre / 5.0f;

        // Feature 1: king safety (edges/corners preferred for king)
        W[1*64 + sq] = (dist_edge < 1.5f) ? 0.8f : -0.3f;

        // Feature 2: pawn rows (rows 1 and 6)
        W[2*64 + sq] = (row == 1 || row == 6) ? 1.0f : 0.0f;

        // Feature 3: piece activity (central 4x4)
        int cr = (row >= 2 && row <= 5);
        int cc = (col >= 2 && col <= 5);
        W[3*64 + sq] = (cr && cc) ? 1.2f : -0.1f;
    }
}

// Aggregation weights [4 x 1] — how much each feature matters
static const float AGG[4] = { 0.5f, 0.2f, 0.15f, 0.35f };

// ---------------------------------------------------------------------------
// Kernel: compute final scalar score from [Nx4] feature scores
// ---------------------------------------------------------------------------
__global__ void k_aggregate_scores(
    const float * __restrict__ feat,  // [N x 4] row-major
    const float * __restrict__ agg,   // [4]
    float       * __restrict__ score, // [N]
    int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int f = 0; f < 4; f++) s += feat[i*4 + f] * agg[f];
    score[i] = s;
}

// ---------------------------------------------------------------------------
// Kernel: extract pawn row signal from intensity grid for cuFFT input
// Each board: 8 floats from row index 1 (rank 2 — white pawn starting row)
// ---------------------------------------------------------------------------
__global__ void k_extract_pawn_row(
    const float * __restrict__ intens, // [N x 64]
    float       * __restrict__ signal, // [N x 8]  (padded to 8 for FFT)
    int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Row 1 = rank 2 (0-indexed), squares [8..15]
    for (int f = 0; f < 8; f++)
        signal[i*8 + f] = intens[i*64 + 1*8 + f];
}

// ---------------------------------------------------------------------------
// Kernel: compute FFT energy from complex output (high-freq bins 3-4)
// cufftComplex output has (8/2+1)=5 complex bins per board
// ---------------------------------------------------------------------------
__global__ void k_fft_energy(
    const cufftComplex * __restrict__ freq, // [N x 5]
    float              * __restrict__ energy,// [N]
    int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N) return;
    float e = 0.f;
    // High frequency bins (3 and 4) indicate fragmented pawn structure
    for (int b = 3; b <= 4; b++) {
        float re = freq[i*5 + b].x;
        float im = freq[i*5 + b].y;
        e += re*re + im*im;
    }
    energy[i] = sqrtf(e);
}

// ---------------------------------------------------------------------------
// evaluator_run
// ---------------------------------------------------------------------------
int evaluator_run(
    const float  *intensities, // [N x 64] host
    const char  **filenames,
    int           N,
    BoardEval    *results)
{
    if (N <= 0) return 1;

    // -----------------------------------------------------------------------
    // 1. cuBLAS: score = intensities [Nx64] * W [64x4] * agg [4x1]
    // -----------------------------------------------------------------------
    cublasHandle_t cublas;
    if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[eval] cublasCreate failed\n"); return 0;
    }

    float *d_intens=NULL, *d_W=NULL, *d_feat=NULL, *d_agg=NULL, *d_score=NULL;
    cudaMalloc(&d_intens, N*64*sizeof(float));
    cudaMalloc(&d_W,      64*4*sizeof(float));
    cudaMalloc(&d_feat,   N*4*sizeof(float));
    cudaMalloc(&d_agg,    4*sizeof(float));
    cudaMalloc(&d_score,  N*sizeof(float));

    float h_W[64*4];
    build_weight_matrix(h_W);

    cudaMemcpy(d_intens, intensities, N*64*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W,      h_W,         64*4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_agg,    AGG,         4*sizeof(float),    cudaMemcpyHostToDevice);

    // cuBLAS is column-major. We want: feat[NxF] = intens[Nx64] * W[64xF]
    // In col-major:  C[FxN] = W^T[Fx64] * intens^T[64xN]
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, F, N, 64, ...)
    float alpha=1.f, beta=0.f;
    cublasSgemm(cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        4, N, 64,
        &alpha,
        d_W,      64,   // W [64 x 4], col-major
        d_intens, 64,   // intens [64 x N] (transposed view), col-major
        &beta,
        d_feat,   4);   // feat [4 x N] col-major = [N x 4] row-major transposed

    // Aggregate [N x 4] feat scores → [N] scalar
    { int threads=256, blocks=(N+255)/256;
      k_aggregate_scores<<<blocks,threads>>>(d_feat, d_agg, d_score, N); }

    // -----------------------------------------------------------------------
    // 2. cuFFT: pawn structure analysis
    // -----------------------------------------------------------------------
    float        *d_pawn_signal=NULL;
    cufftComplex *d_pawn_freq=NULL;
    float        *d_pawn_energy=NULL;

    cudaMalloc(&d_pawn_signal, N*8*sizeof(float));
    cudaMalloc(&d_pawn_freq,   N*5*sizeof(cufftComplex));
    cudaMalloc(&d_pawn_energy, N*sizeof(float));

    { int t=256, b=(N+255)/256;
      k_extract_pawn_row<<<b,t>>>(d_intens, d_pawn_signal, N); }

    // Batch R2C FFT: N signals each of length 8 → N * 5 complex outputs
    cufftHandle fft_plan;
    int fft_len = 8;
    cufftPlanMany(&fft_plan, 1, &fft_len,
                  NULL, 1, fft_len,   // input stride/dist
                  NULL, 1, fft_len/2+1,// output stride/dist
                  CUFFT_R2C, N);
    cufftExecR2C(fft_plan, d_pawn_signal, d_pawn_freq);
    cufftDestroy(fft_plan);

    { int t=256, b=(N+255)/256;
      k_fft_energy<<<b,t>>>(d_pawn_freq, d_pawn_energy, N); }

    cudaDeviceSynchronize();

    // -----------------------------------------------------------------------
    // 3. Thrust: sort boards by score descending, assign ranks
    // -----------------------------------------------------------------------
    thrust::device_vector<float> dv_score(d_score, d_score+N);
    thrust::device_vector<int>   dv_rank(N);
    thrust::sequence(dv_rank.begin(), dv_rank.end(), 0); // 0..N-1

    // Sort indices by score descending
    thrust::sort_by_key(dv_score.begin(), dv_score.end(), dv_rank.begin(),
                        thrust::greater<float>());

    // -----------------------------------------------------------------------
    // Download results
    // -----------------------------------------------------------------------
    float *h_scores = (float*)malloc(N*sizeof(float));
    float *h_energy = (float*)malloc(N*sizeof(float));
    int   *h_ranks  = (int*)  malloc(N*sizeof(int));

    // After sort, dv_score is sorted and dv_rank holds original indices
    // We need: for each original board i, what is its rank?
    thrust::host_vector<float> hv_score(dv_score);
    thrust::host_vector<int>   hv_rank(dv_rank);

    // Build reverse lookup: orig_idx → rank
    int *rank_of = (int*)malloc(N*sizeof(int));
    for (int r=0; r<N; r++) rank_of[hv_rank[r]] = r;

    cudaMemcpy(h_scores, d_score,      N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_energy, d_pawn_energy,N*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<N; i++) {
        results[i].score          = h_scores[i];
        results[i].material       = h_scores[i] * 2.0f; // proxy
        results[i].pawn_structure = h_energy[i];
        results[i].rank           = rank_of[i];
        strncpy(results[i].filename, filenames[i], 255);
        results[i].filename[255]  = '\0';
    }

    free(h_scores); free(h_energy); free(h_ranks); free(rank_of);

    cublasDestroy(cublas);
    cudaFree(d_intens); cudaFree(d_W);    cudaFree(d_feat);
    cudaFree(d_agg);    cudaFree(d_score);
    cudaFree(d_pawn_signal); cudaFree(d_pawn_freq); cudaFree(d_pawn_energy);
    return 1;
}

int evaluator_save_csv(const char *path, const BoardEval *results, int N) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fprintf(f, "rank,filename,score,material_proxy,pawn_fft_energy\n");
    // Print in rank order
    // Simple: allocate sorted index array
    int *order = (int*)malloc(N*sizeof(int));
    for (int i=0; i<N; i++) order[i]=i;
    // Bubble sort by rank (N is small)
    for (int i=0; i<N-1; i++)
        for (int j=0; j<N-1-i; j++)
            if (results[order[j]].rank > results[order[j+1]].rank) {
                int t=order[j]; order[j]=order[j+1]; order[j+1]=t;
            }
    for (int i=0; i<N; i++) {
        const BoardEval *r = &results[order[i]];
        // strip directory from filename
        const char *bn = strrchr(r->filename, '/');
        bn = bn ? bn+1 : r->filename;
        fprintf(f, "%d,%s,%.4f,%.4f,%.4f\n",
                r->rank+1, bn, r->score, r->material, r->pawn_structure);
    }
    free(order);
    fclose(f);
    return 1;
}
