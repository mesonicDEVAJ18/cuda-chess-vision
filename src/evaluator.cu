// evaluator.cu — Stage 3: GPU batch board evaluation
//
// Three GPU libraries:
//   cuBLAS  : batch matrix multiply [N×64] × [64×4] → spatial feature scores
//   cuFFT   : batch R2C FFT on white pawn file distributions → pawn structure score
//   Thrust  : parallel sort_by_key descending → rank assignment
//
// Plus custom kernel k_pst_score using standard piece-square tables
// stored in __constant__ memory — the primary chess evaluation.
//
// Input:  float[N×64] board states from board_to_floats()
//         Piece encoding: +1.0=wP +3.0=wN +3.3=wB +5.0=wR +9.0=wQ +20.0=wK
//                         negated for black pieces, 0=empty

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "evaluator.h"

// ---------------------------------------------------------------------------
// Piece-square tables (centipawns, white perspective)
// Index: sq = rank*8 + file, rank0 = rank-1 (white's back rank)
// For black pieces: mirror rank (7-rank)*8+file, then negate
// ---------------------------------------------------------------------------
__constant__ float d_pst[6][64] = {
    { // 0 = Pawn
       0,  0,  0,  0,  0,  0,  0,  0,
       5, 10, 10,-20,-20, 10, 10,  5,
       5, -5,-10,  0,  0,-10, -5,  5,
       0,  0,  0, 20, 20,  0,  0,  0,
       5,  5, 10, 25, 25, 10,  5,  5,
      10, 10, 20, 30, 30, 20, 10, 10,
      50, 50, 50, 50, 50, 50, 50, 50,
       0,  0,  0,  0,  0,  0,  0,  0 },
    { // 1 = Knight
     -50,-40,-30,-30,-30,-30,-40,-50,
     -40,-20,  0,  5,  5,  0,-20,-40,
     -30,  5, 10, 15, 15, 10,  5,-30,
     -30,  0, 15, 20, 20, 15,  0,-30,
     -30,  5, 15, 20, 20, 15,  5,-30,
     -30,  0, 10, 15, 15, 10,  0,-30,
     -40,-20,  0,  0,  0,  0,-20,-40,
     -50,-40,-30,-30,-30,-30,-40,-50 },
    { // 2 = Bishop
     -20,-10,-10,-10,-10,-10,-10,-20,
     -10,  5,  0,  0,  0,  0,  5,-10,
     -10, 10, 10, 10, 10, 10, 10,-10,
     -10,  0, 10, 10, 10, 10,  0,-10,
     -10,  5,  5, 10, 10,  5,  5,-10,
     -10,  0,  5, 10, 10,  5,  0,-10,
     -10,  0,  0,  0,  0,  0,  0,-10,
     -20,-10,-10,-10,-10,-10,-10,-20 },
    { // 3 = Rook
       0,  0,  0,  5,  5,  0,  0,  0,
      -5,  0,  0,  0,  0,  0,  0, -5,
      -5,  0,  0,  0,  0,  0,  0, -5,
      -5,  0,  0,  0,  0,  0,  0, -5,
      -5,  0,  0,  0,  0,  0,  0, -5,
      -5,  0,  0,  0,  0,  0,  0, -5,
       5, 10, 10, 10, 10, 10, 10,  5,
       0,  0,  0,  0,  0,  0,  0,  0 },
    { // 4 = Queen
     -20,-10,-10, -5, -5,-10,-10,-20,
     -10,  0,  5,  0,  0,  0,  0,-10,
     -10,  5,  5,  5,  5,  5,  0,-10,
       0,  0,  5,  5,  5,  5,  0, -5,
      -5,  0,  5,  5,  5,  5,  0, -5,
     -10,  0,  5,  5,  5,  5,  0,-10,
     -10,  0,  0,  0,  0,  0,  0,-10,
     -20,-10,-10, -5, -5,-10,-10,-20 },
    { // 5 = King (middlegame safety)
      20, 30, 10,  0,  0, 10, 30, 20,
      20, 20,  0,  0,  0,  0, 20, 20,
     -10,-20,-20,-20,-20,-20,-20,-10,
     -20,-30,-30,-40,-40,-30,-30,-20,
     -30,-40,-40,-50,-50,-40,-40,-30,
     -30,-40,-40,-50,-50,-40,-40,-30,
     -30,-40,-40,-50,-50,-40,-40,-30,
     -30,-40,-40,-50,-50,-40,-40,-30 }
};

// ---------------------------------------------------------------------------
// Kernel 1 — per-piece PST evaluation using constant-memory tables
// Grid: N blocks (one per board), 64 threads (one per square)
// ---------------------------------------------------------------------------
__global__ void k_pst_score(
    const float * __restrict__ boards,   // [N × 64]
    float       * __restrict__ pst_out,  // [N]   (zero-initialised before call)
    int N)
{
    int board = blockIdx.x;
    int sq    = threadIdx.x;
    if (board >= N || sq >= 64) return;

    float val = boards[board * 64 + sq];
    if (fabsf(val) < 0.05f) return;   // empty square

    int sign = (val > 0.f) ? 1 : -1;
    float av = fabsf(val);

    // Identify piece type from encoding
    int pt;
    if      (av < 1.5f)  pt = 0;   // pawn  (1.0)
    else if (av < 3.15f) pt = 1;   // knight(3.0)
    else if (av < 4.0f)  pt = 2;   // bishop(3.3)
    else if (av < 7.0f)  pt = 3;   // rook  (5.0)
    else if (av < 15.f)  pt = 4;   // queen (9.0)
    else                  pt = 5;   // king  (20.0)

    // Mirror rank for black pieces
    int rank = sq / 8, file = sq % 8;
    int pst_sq = (sign > 0) ? sq : (7 - rank) * 8 + file;

    // Normalise to roughly [-1, 1] range by dividing by 50 centipawns
    float contribution = (float)sign * d_pst[pt][pst_sq] / 50.f;
    atomicAdd(&pst_out[board], contribution);
}

// ---------------------------------------------------------------------------
// Kernel 2 — aggregate cuBLAS feature matrix [N×4] → scalar [N]
// ---------------------------------------------------------------------------
__global__ void k_aggregate(
    const float * __restrict__ feat, // [N × 4] row-major
    const float * __restrict__ agg,  // [4]
    float       * __restrict__ out,  // [N]
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.f;
    for (int f = 0; f < 4; f++) s += feat[i * 4 + f] * agg[f];
    out[i] = s;
}

// ---------------------------------------------------------------------------
// Kernel 3 — extract white pawn file distribution [N × 8]
// ---------------------------------------------------------------------------
__global__ void k_pawn_files(
    const float * __restrict__ boards, // [N × 64]
    float       * __restrict__ files,  // [N × 8]
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float *f = files + i * 8;
    for (int k = 0; k < 8; k++) f[k] = 0.f;
    const float *b = boards + i * 64;
    for (int sq = 0; sq < 64; sq++)
        if (fabsf(b[sq] - 1.0f) < 0.05f)  // white pawn = 1.0
            f[sq % 8] += 1.f;
}

// ---------------------------------------------------------------------------
// Kernel 4 — FFT high-frequency energy (pawn fragmentation)
// ---------------------------------------------------------------------------
__global__ void k_fft_energy(
    const cufftComplex * __restrict__ freq,   // [N × 5]  (8→5 complex bins)
    float              * __restrict__ energy, // [N]
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float e = 0.f;
    for (int b = 3; b <= 4; b++) {   // high-frequency bins 3 and 4
        float re = freq[i*5 + b].x, im = freq[i*5 + b].y;
        e += re*re + im*im;
    }
    energy[i] = sqrtf(e);
}

// ---------------------------------------------------------------------------
// Kernel 5 — combine all sub-scores → final score
// ---------------------------------------------------------------------------
__global__ void k_combine_scores(
    const float * __restrict__ pst,      // [N] PST score
    const float * __restrict__ spatial,  // [N] cuBLAS spatial score
    const float * __restrict__ pawn_fft, // [N] pawn fragmentation energy
    float       * __restrict__ total,    // [N]
    float       * __restrict__ material, // [N] (material only, for CSV)
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // spatial[i][0] (col 0, all-ones weight) = raw material balance
    // We'll compute material separately in host after cuBLAS download.
    // total = weighted combination: PST dominates, spatial adds spatial sense,
    //         pawn fragmentation penalty (higher energy = worse structure)
    total[i] = pst[i] * 0.6f + spatial[i] * 0.4f - pawn_fft[i] * 0.015f;
}

// ---------------------------------------------------------------------------
// Build cuBLAS spatial weight matrix [64 × 4] column-major
// Col 0: uniform (material balance)
// Col 1: centre bonus  (d4/d5/e4/e5 = max)
// Col 2: non-edge bonus
// Col 3: active-square bonus (inner 4×4)
// ---------------------------------------------------------------------------
static void build_spatial_weights(float *W)
{
    for (int sq = 0; sq < 64; sq++) {
        int rank = sq / 8, file = sq % 8;
        float dr = rank - 3.5f, dc = file - 3.5f;
        float dist  = sqrtf(dr*dr + dc*dc);
        float edge  = fminf(fminf((float)rank, (float)(7-rank)),
                            fminf((float)file,  (float)(7-file)));
        int inner = (rank>=2 && rank<=5 && file>=2 && file<=5);

        W[0*64 + sq] = 1.0f;                              // material
        W[1*64 + sq] = fmaxf(0.0f, 1.f - dist / 5.f);   // centre
        W[2*64 + sq] = fminf(1.f, edge / 2.f + 0.3f);   // non-edge
        W[3*64 + sq] = inner ? 1.0f : 0.1f;              // activity
    }
}

static const float AGG[4] = { 0.5f, 0.2f, 0.15f, 0.15f };

// ---------------------------------------------------------------------------
// evaluator_run — host entry point
// ---------------------------------------------------------------------------
int evaluator_run(
    const float  *board_states,  // host [N×64]
    const char  **filenames,
    int           N,
    BoardEval    *results)
{
    if (N <= 0) return 1;

    // ---- Device allocations ----
    float        *d_boards=NULL, *d_W=NULL, *d_feat=NULL, *d_agg=NULL;
    float        *d_spatial=NULL, *d_pst_scores=NULL, *d_total=NULL;
    float        *d_pawn_files=NULL, *d_pawn_energy=NULL;
    cufftComplex *d_pawn_freq=NULL;

    size_t szN  = (size_t)N * sizeof(float);

#define ALLOC(ptr,bytes) if(cudaMalloc(&(ptr),(bytes))!=cudaSuccess) goto fail
    ALLOC(d_boards,      (size_t)N*64*sizeof(float));
    ALLOC(d_W,           64*4*sizeof(float));
    ALLOC(d_feat,        (size_t)N*4*sizeof(float));
    ALLOC(d_agg,         4*sizeof(float));
    ALLOC(d_spatial,     szN);
    ALLOC(d_pst_scores,  szN);
    ALLOC(d_total,       szN);
    ALLOC(d_pawn_files,  (size_t)N*8*sizeof(float));
    ALLOC(d_pawn_freq,   (size_t)N*5*sizeof(cufftComplex));
    ALLOC(d_pawn_energy, szN);
#undef ALLOC

    // Upload boards
    cudaMemcpy(d_boards, board_states, (size_t)N*64*sizeof(float),
               cudaMemcpyHostToDevice);

    // ---- PST kernel ----
    cudaMemset(d_pst_scores, 0, szN);
    k_pst_score<<<N, 64>>>(d_boards, d_pst_scores, N);

    // ---- cuBLAS: spatial features ----
    {
        float h_W[64*4];
        build_spatial_weights(h_W);
        cudaMemcpy(d_W,   h_W, 64*4*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_agg, AGG, 4*sizeof(float),    cudaMemcpyHostToDevice);

        cublasHandle_t cb;
        if (cublasCreate(&cb) != CUBLAS_STATUS_SUCCESS) goto fail;

        float alpha=1.f, beta=0.f;
        // feat[N×4] = boards[N×64] × W[64×4]
        // cuBLAS col-major: C[4×N] = W^T[4×64] × boards^T[64×N]
        cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N,
                    4, N, 64,
                    &alpha, d_W, 64, d_boards, 64,
                    &beta,  d_feat, 4);
        cublasDestroy(cb);

        int thr=256, blk=(N+255)/256;
        k_aggregate<<<blk,thr>>>(d_feat, d_agg, d_spatial, N);
    }

    // ---- cuFFT: pawn structure ----
    {
        int thr=256, blk=(N+255)/256;
        k_pawn_files<<<blk,thr>>>(d_boards, d_pawn_files, N);

        cufftHandle plan;
        int fft_len = 8;
        cufftPlanMany(&plan, 1, &fft_len,
                      NULL, 1, fft_len,
                      NULL, 1, fft_len/2+1,
                      CUFFT_R2C, N);
        cufftExecR2C(plan, d_pawn_files, d_pawn_freq);
        cufftDestroy(plan);

        k_fft_energy<<<blk,thr>>>(d_pawn_freq, d_pawn_energy, N);
    }

    cudaDeviceSynchronize();

    // ---- Combine scores ----
    {
        int thr=256, blk=(N+255)/256;
        k_combine_scores<<<blk,thr>>>(
            d_pst_scores, d_spatial, d_pawn_energy, d_total, NULL, N);
    }
    cudaDeviceSynchronize();

    // ---- Thrust sort + rank ----
    {
        thrust::device_vector<float> dv_total(d_total, d_total + N);
        thrust::device_vector<int>   dv_idx(N);
        thrust::sequence(dv_idx.begin(), dv_idx.end(), 0);
        thrust::sort_by_key(dv_total.begin(), dv_total.end(),
                            dv_idx.begin(), thrust::greater<float>());

        thrust::host_vector<float> hv_total(dv_total);
        thrust::host_vector<int>   hv_idx(dv_idx);

        // Build reverse map: original index → rank
        int *rank_of = (int *)malloc(N * sizeof(int));
        for (int r = 0; r < N; r++) rank_of[hv_idx[r]] = r;

        // Download sub-scores
        float *h_pst     = (float *)malloc(szN);
        float *h_pawn    = (float *)malloc(szN);
        float *h_total   = (float *)malloc(szN);
        float *h_feat    = (float *)malloc((size_t)N*4*sizeof(float));

        cudaMemcpy(h_pst,   d_pst_scores, szN, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pawn,  d_pawn_energy,szN, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_total, d_total,      szN, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_feat,  d_feat, (size_t)N*4*sizeof(float),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < N; i++) {
            // feat is col-major: feat[col][board] = h_feat[col*N + i]
            // col 0 = material (all-ones spatial weight → raw piece-value sum)
            results[i].material       = h_feat[0*N + i];
            results[i].positional     = h_pst[i];
            results[i].pawn_structure = h_pawn[i];
            results[i].score          = h_total[i];   // pre-combine score
            results[i].rank           = rank_of[i];
            strncpy(results[i].filename, filenames[i], 255);
            results[i].filename[255]  = '\0';
        }

        free(h_pst); free(h_pawn); free(h_total); free(h_feat); free(rank_of);
    }

    cudaFree(d_boards);   cudaFree(d_W);       cudaFree(d_feat);
    cudaFree(d_agg);      cudaFree(d_spatial);  cudaFree(d_pst_scores);
    cudaFree(d_total);    cudaFree(d_pawn_files);
    cudaFree(d_pawn_freq);cudaFree(d_pawn_energy);
    return 1;

fail:
    fprintf(stderr, "[evaluator] CUDA error: %s\n",
            cudaGetErrorString(cudaGetLastError()));
    cudaFree(d_boards);   cudaFree(d_W);       cudaFree(d_feat);
    cudaFree(d_agg);      cudaFree(d_spatial);  cudaFree(d_pst_scores);
    cudaFree(d_total);    cudaFree(d_pawn_files);
    cudaFree(d_pawn_freq);cudaFree(d_pawn_energy);
    return 0;
}

// ---------------------------------------------------------------------------
// evaluator_save_csv
// ---------------------------------------------------------------------------
int evaluator_save_csv(const char *path, const BoardEval *results, int N)
{
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fprintf(f, "rank,filename,score,material,positional,pawn_fft_energy\n");

    // Sort by rank for output
    int *order = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) order[i] = i;
    for (int i = 0; i < N-1; i++)
        for (int j = 0; j < N-1-i; j++)
            if (results[order[j]].rank > results[order[j+1]].rank) {
                int tmp = order[j]; order[j] = order[j+1]; order[j+1] = tmp;
            }

    for (int i = 0; i < N; i++) {
        const BoardEval *r = &results[order[i]];
        const char *bn = strrchr(r->filename, '/');
        bn = bn ? bn+1 : r->filename;
        fprintf(f, "%d,%s,%.4f,%.4f,%.4f,%.4f\n",
                r->rank+1, bn, r->score, r->material,
                r->positional, r->pawn_structure);
    }
    free(order);
    fclose(f);
    return 1;
}
