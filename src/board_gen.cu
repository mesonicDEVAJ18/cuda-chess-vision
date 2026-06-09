// board_gen.cu — Stage 1: GPU chess position generation + pixel-art renderer
//
// GPU (cuRAND):  k_rand_init + k_generate_boards
//   Each CUDA thread generates one full legal board:
//   kings placed first with separation constraint, then random piece counts
//   for each piece type with pawn-rank restrictions.
//
// CPU (libpng):  board_save_png()
//   Draws 6 distinct piece silhouettes (pawn/knight/bishop/rook/queen/king)
//   in white and black variants onto a 512×512 RGB checkerboard.

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <png.h>

#include "board_gen.h"

// ---------------------------------------------------------------------------
// Square colours (Lichess palette)
// ---------------------------------------------------------------------------
#define LIGHT_R 240, 217, 181
#define DARK_R  181, 136,  99

// ---------------------------------------------------------------------------
// Pixel helpers
// ---------------------------------------------------------------------------
static void set_pixel(uint8_t *rgb, int W, int x, int y,
                      uint8_t r, uint8_t g, uint8_t b)
{
    if (x < 0 || x >= W || y < 0 || y >= W) return;
    int i = (y * W + x) * 3;
    rgb[i] = r; rgb[i+1] = g; rgb[i+2] = b;
}

static void fill_ellipse(uint8_t *rgb, int W,
                         int cx, int cy, int rx, int ry,
                         uint8_t r, uint8_t g, uint8_t b)
{
    for (int dy = -ry; dy <= ry; dy++)
        for (int dx = -rx; dx <= rx; dx++) {
            float nx = (float)dx / rx, ny = (float)dy / ry;
            if (nx*nx + ny*ny <= 1.0f)
                set_pixel(rgb, W, cx+dx, cy+dy, r, g, b);
        }
}

static void fill_rect(uint8_t *rgb, int W,
                      int x0, int y0, int x1, int y1,
                      uint8_t r, uint8_t g, uint8_t b)
{
    for (int y = y0; y <= y1; y++)
        for (int x = x0; x <= x1; x++)
            set_pixel(rgb, W, x, y, r, g, b);
}

static void fill_trapezoid(uint8_t *rgb, int W,
                           int cx, int y0, int y1,
                           int top_w, int bot_w,
                           uint8_t r, uint8_t g, uint8_t b)
{
    if (y1 < y0) return;
    for (int y = y0; y <= y1; y++) {
        float t  = (y1 > y0) ? (float)(y - y0) / (y1 - y0) : 0.f;
        int half = (int)((top_w + t * (bot_w - top_w)) * 0.5f);
        for (int x = cx - half; x <= cx + half; x++)
            set_pixel(rgb, W, x, y, r, g, b);
    }
}

// ---------------------------------------------------------------------------
// Draw one piece silhouette at square centre (cx, cy).
// piece_type: 1=pawn 2=knight 3=bishop 4=rook 5=queen 6=king
// white: 1 for white piece, 0 for black piece
// ---------------------------------------------------------------------------
static void draw_piece(uint8_t *rgb, int W,
                       int piece_type, int white,
                       int cx, int cy)
{
    uint8_t fr, fg, fb;   // fill colour
    uint8_t or_, og, ob;  // outline colour
    if (white) {
        fr=252; fg=252; fb=252;
        or_=55; og=55; ob=55;
    } else {
        fr=22; fg=22; fb=22;
        or_=185; og=185; ob=185;
    }

    switch (piece_type) {
    // ------------------------------------------------------------------
    case 1: // PAWN
        // outline head
        fill_ellipse(rgb,W, cx,cy-13, 12,12, or_,og,ob);
        // fill head
        fill_ellipse(rgb,W, cx,cy-13, 10,10, fr,fg,fb);
        // outline stem
        fill_rect(rgb,W, cx-6,cy-3, cx+6,cy+4, or_,og,ob);
        // fill stem
        fill_rect(rgb,W, cx-4,cy-2, cx+4,cy+3, fr,fg,fb);
        // outline body
        fill_trapezoid(rgb,W, cx, cy+4,cy+15, 18,28, or_,og,ob);
        // fill body
        fill_trapezoid(rgb,W, cx, cy+5,cy+14, 16,26, fr,fg,fb);
        // outline base
        fill_rect(rgb,W, cx-16,cy+15, cx+16,cy+21, or_,og,ob);
        // fill base
        fill_rect(rgb,W, cx-15,cy+16, cx+15,cy+20, fr,fg,fb);
        break;

    // ------------------------------------------------------------------
    case 2: // KNIGHT
        // outline body oval
        fill_ellipse(rgb,W, cx-2,cy-8, 15,17, or_,og,ob);
        // fill body oval
        fill_ellipse(rgb,W, cx-2,cy-8, 13,15, fr,fg,fb);
        // outline snout
        fill_ellipse(rgb,W, cx+9,cy-18, 10,7, or_,og,ob);
        // fill snout
        fill_ellipse(rgb,W, cx+9,cy-18, 8,5, fr,fg,fb);
        // nostril dot (fill colour of opposite for contrast)
        fill_ellipse(rgb,W, cx+13,cy-17, 2,2, or_,og,ob);
        // outline base
        fill_rect(rgb,W, cx-15,cy+9, cx+15,cy+21, or_,og,ob);
        // fill base
        fill_rect(rgb,W, cx-14,cy+10, cx+14,cy+20, fr,fg,fb);
        break;

    // ------------------------------------------------------------------
    case 3: // BISHOP
        // top ball outline + fill
        fill_ellipse(rgb,W, cx,cy-23, 6,6, or_,og,ob);
        fill_ellipse(rgb,W, cx,cy-23, 4,4, fr,fg,fb);
        // mitre outline + fill (tall narrow ellipse)
        fill_ellipse(rgb,W, cx,cy-9, 10,15, or_,og,ob);
        fill_ellipse(rgb,W, cx,cy-9, 8,13, fr,fg,fb);
        // body outline + fill
        fill_trapezoid(rgb,W, cx, cy+6,cy+14, 16,26, or_,og,ob);
        fill_trapezoid(rgb,W, cx, cy+7,cy+13, 14,24, fr,fg,fb);
        // base outline + fill
        fill_rect(rgb,W, cx-15,cy+14, cx+15,cy+21, or_,og,ob);
        fill_rect(rgb,W, cx-14,cy+15, cx+14,cy+20, fr,fg,fb);
        break;

    // ------------------------------------------------------------------
    case 4: // ROOK
        // 3 battlements (outline then fill)
        fill_rect(rgb,W, cx-13,cy-24, cx-6, cy-15, or_,og,ob);
        fill_rect(rgb,W, cx-12,cy-23, cx-7, cy-16, fr,fg,fb);
        fill_rect(rgb,W, cx-4, cy-24, cx+4, cy-15, or_,og,ob);
        fill_rect(rgb,W, cx-3, cy-23, cx+3, cy-16, fr,fg,fb);
        fill_rect(rgb,W, cx+6, cy-24, cx+13,cy-15, or_,og,ob);
        fill_rect(rgb,W, cx+7, cy-23, cx+12,cy-16, fr,fg,fb);
        // column body outline + fill
        fill_rect(rgb,W, cx-13,cy-17, cx+13,cy+12, or_,og,ob);
        fill_rect(rgb,W, cx-12,cy-16, cx+12,cy+11, fr,fg,fb);
        // base outline + fill
        fill_rect(rgb,W, cx-16,cy+12, cx+16,cy+21, or_,og,ob);
        fill_rect(rgb,W, cx-15,cy+13, cx+15,cy+20, fr,fg,fb);
        break;

    // ------------------------------------------------------------------
    case 5: // QUEEN
        // 3 crown circles: left, top, right
        fill_ellipse(rgb,W, cx-11,cy-23, 7,7, or_,og,ob);
        fill_ellipse(rgb,W, cx-11,cy-23, 5,5, fr,fg,fb);
        fill_ellipse(rgb,W, cx,   cy-26, 7,7, or_,og,ob);
        fill_ellipse(rgb,W, cx,   cy-26, 5,5, fr,fg,fb);
        fill_ellipse(rgb,W, cx+11,cy-23, 7,7, or_,og,ob);
        fill_ellipse(rgb,W, cx+11,cy-23, 5,5, fr,fg,fb);
        // body outline + fill
        fill_trapezoid(rgb,W, cx, cy-18,cy+11, 24,30, or_,og,ob);
        fill_trapezoid(rgb,W, cx, cy-17,cy+10, 22,28, fr,fg,fb);
        // waist ring
        fill_rect(rgb,W, cx-13,cy-5, cx+13,cy-1, or_,og,ob);
        fill_rect(rgb,W, cx-12,cy-4, cx+12,cy-2, fr,fg,fb);
        // base outline + fill
        fill_rect(rgb,W, cx-16,cy+11, cx+16,cy+21, or_,og,ob);
        fill_rect(rgb,W, cx-15,cy+12, cx+15,cy+20, fr,fg,fb);
        break;

    // ------------------------------------------------------------------
    case 6: // KING
        // cross: vertical bar
        fill_rect(rgb,W, cx-3,cy-28, cx+3,cy-14, or_,og,ob);
        fill_rect(rgb,W, cx-2,cy-27, cx+2,cy-15, fr,fg,fb);
        // cross: horizontal bar
        fill_rect(rgb,W, cx-9,cy-24, cx+9,cy-19, or_,og,ob);
        fill_rect(rgb,W, cx-8,cy-23, cx+8,cy-20, fr,fg,fb);
        // body outline + fill
        fill_trapezoid(rgb,W, cx, cy-14,cy+11, 22,30, or_,og,ob);
        fill_trapezoid(rgb,W, cx, cy-13,cy+10, 20,28, fr,fg,fb);
        // waist ring
        fill_rect(rgb,W, cx-14,cy-3, cx+14,cy+1, or_,og,ob);
        fill_rect(rgb,W, cx-13,cy-2, cx+13,cy,   fr,fg,fb);
        // base outline + fill
        fill_rect(rgb,W, cx-16,cy+11, cx+16,cy+21, or_,og,ob);
        fill_rect(rgb,W, cx-15,cy+12, cx+15,cy+20, fr,fg,fb);
        break;
    }
}

// ---------------------------------------------------------------------------
// board_save_png — render ChessBoard → 512×512 RGB PNG
// ---------------------------------------------------------------------------
int board_save_png(const char *path, const ChessBoard *b, int size)
{
    int sq_size = size / 8;
    uint8_t *rgb = (uint8_t *)malloc(size * size * 3);
    if (!rgb) return 0;

    // Draw squares
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            // rank 7 = top of screen (black's back rank)
            int screen_row = 7 - rank;
            int sx = file   * sq_size;
            int sy = screen_row * sq_size;
            int light = ((file + rank) % 2 == 0);
            uint8_t sr = light ? 240 : 181;
            uint8_t sg = light ? 217 : 136;
            uint8_t sb = light ? 181 :  99;
            for (int py = sy; py < sy + sq_size; py++)
                for (int px = sx; px < sx + sq_size; px++)
                    set_pixel(rgb, size, px, py, sr, sg, sb);

            // Draw piece if present
            int8_t code = b->squares[rank * 8 + file];
            if (code != 0) {
                int cx = sx + sq_size / 2;
                int cy = sy + sq_size / 2;
                draw_piece(rgb, size,
                           (int)(code > 0 ? code : -code),
                           (code > 0) ? 1 : 0,
                           cx, cy);
            }
        }
    }

    // Draw rank/file labels (thin border)
    // (skipped for clean look — labels can be added in visualize.py)

    // Save via libpng
    FILE *fp = fopen(path, "wb");
    if (!fp) { free(rgb); return 0; }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                              NULL, NULL, NULL);
    png_infop   info = png_create_info_struct(png);
    if (!png || !info) {
        if (png) png_destroy_write_struct(&png, NULL);
        fclose(fp); free(rgb); return 0;
    }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp); free(rgb); return 0;
    }
    png_init_io(png, fp);
    png_set_IHDR(png, info, size, size, 8,
                 PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    for (int y = 0; y < size; y++)
        png_write_row(png, rgb + y * size * 3);
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    free(rgb);
    return 1;
}

// ---------------------------------------------------------------------------
// board_to_floats — convert piece codes to evaluation float values
// ---------------------------------------------------------------------------
void board_to_floats(const ChessBoard *b, float out[64])
{
    static const float piece_val[7] = {0.f, 1.f, 3.f, 3.3f, 5.f, 9.f, 20.f};
    for (int i = 0; i < 64; i++) {
        int8_t c = b->squares[i];
        if (c == 0) { out[i] = 0.f; continue; }
        int abs_c = c > 0 ? c : -c;
        if (abs_c > 6) abs_c = 6;
        out[i] = (c > 0) ? piece_val[abs_c] : -piece_val[abs_c];
    }
}

// ---------------------------------------------------------------------------
// CUDA kernels — GPU position generation
// ---------------------------------------------------------------------------

__global__ void k_rand_init(curandState *states, unsigned long long seed, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) curand_init(seed + (unsigned long long)i * 6364136223846793005ULL,
                           i, 0, &states[i]);
}

__device__ static int kings_adjacent(int a, int b)
{
    int ar = a/8, af = a%8, br = b/8, bf = b%8;
    int dr = ar-br, df = af-bf;
    return (dr>=-1 && dr<=1 && df>=-1 && df<=1);
}

__device__ static int try_place(int8_t *sq, int8_t piece,
                                curandState *rng, int tries)
{
    for (int t = 0; t < tries; t++) {
        int pos = (int)(curand_uniform(rng) * 64.f) % 64;
        if (sq[pos] == 0) { sq[pos] = piece; return 1; }
    }
    return 0;
}

__device__ static int try_place_ranks(int8_t *sq, int8_t piece,
                                      int r0, int r1,
                                      curandState *rng, int tries)
{
    for (int t = 0; t < tries; t++) {
        int rank = r0 + (int)(curand_uniform(rng) * (float)(r1 - r0 + 1)) % (r1 - r0 + 1);
        int file = (int)(curand_uniform(rng) * 8.f) % 8;
        int pos  = rank * 8 + file;
        if (sq[pos] == 0) { sq[pos] = piece; return 1; }
    }
    return 0;
}

__global__ void k_generate_boards(int8_t *boards, curandState *states, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState rng = states[i];
    int8_t *sq = boards + i * 64;

    // Clear
    for (int j = 0; j < 64; j++) sq[j] = 0;

    // --- White king: ranks 0-2 ---
    int wk = -1;
    for (int t = 0; t < 64 && wk < 0; t++) {
        int r = (int)(curand_uniform(&rng) * 3.f) % 3;
        int f = (int)(curand_uniform(&rng) * 8.f) % 8;
        int p = r*8 + f;
        if (sq[p] == 0) { sq[p] = 6; wk = p; }
    }
    if (wk < 0) { sq[4] = 6; wk = 4; } // fallback e1

    // --- Black king: ranks 5-7, not adjacent to white king ---
    int bk = -1;
    for (int t = 0; t < 128 && bk < 0; t++) {
        int r = 5 + (int)(curand_uniform(&rng) * 3.f) % 3;
        int f = (int)(curand_uniform(&rng) * 8.f) % 8;
        int p = r*8 + f;
        if (sq[p] == 0 && !kings_adjacent(wk, p)) {
            sq[p] = -6; bk = p;
        }
    }
    if (bk < 0) {
        // guaranteed fallback: rank 7 opposite file from white king
        int f = (wk % 8 + 4) % 8;
        sq[7*8 + f] = -6; bk = 7*8 + f;
    }

    // --- Piece counts (random) ---
    int nwQ = (int)(curand_uniform(&rng) * 2.f);   // 0-1
    int nbQ = (int)(curand_uniform(&rng) * 2.f);
    int nwR = (int)(curand_uniform(&rng) * 3.f);   // 0-2
    int nbR = (int)(curand_uniform(&rng) * 3.f);
    int nwB = (int)(curand_uniform(&rng) * 3.f);
    int nbB = (int)(curand_uniform(&rng) * 3.f);
    int nwN = (int)(curand_uniform(&rng) * 3.f);
    int nbN = (int)(curand_uniform(&rng) * 3.f);
    int nwP = 2 + (int)(curand_uniform(&rng) * 7.f); // 2-8
    int nbP = 2 + (int)(curand_uniform(&rng) * 7.f);

    // Queens
    for (int j=0;j<nwQ;j++) try_place(sq,  5, &rng, 50);
    for (int j=0;j<nbQ;j++) try_place(sq, -5, &rng, 50);
    // Rooks
    for (int j=0;j<nwR;j++) try_place(sq,  4, &rng, 50);
    for (int j=0;j<nbR;j++) try_place(sq, -4, &rng, 50);
    // Bishops
    for (int j=0;j<nwB;j++) try_place(sq,  3, &rng, 50);
    for (int j=0;j<nbB;j++) try_place(sq, -3, &rng, 50);
    // Knights
    for (int j=0;j<nwN;j++) try_place(sq,  2, &rng, 50);
    for (int j=0;j<nbN;j++) try_place(sq, -2, &rng, 50);
    // Pawns: ranks 1-6 only
    for (int j=0;j<nwP;j++) try_place_ranks(sq,  1, 1, 6, &rng, 60);
    for (int j=0;j<nbP;j++) try_place_ranks(sq, -1, 1, 6, &rng, 60);

    states[i] = rng;
}

// ---------------------------------------------------------------------------
// boards_generate_gpu — host entry point
// ---------------------------------------------------------------------------
ChessBoard *boards_generate_gpu(int N, unsigned long long seed)
{
    if (N <= 0) return NULL;

    curandState *d_states = NULL;
    int8_t      *d_boards = NULL;

    if (cudaMalloc(&d_states, N * sizeof(curandState)) != cudaSuccess) goto fail;
    if (cudaMalloc(&d_boards, N * 64 * sizeof(int8_t))  != cudaSuccess) goto fail;

    {
        int threads = 128, blocks = (N + threads - 1) / threads;
        k_rand_init<<<blocks, threads>>>(d_states, seed, N);
        if (cudaGetLastError() != cudaSuccess) goto fail;

        k_generate_boards<<<blocks, threads>>>(d_boards, d_states, N);
        if (cudaGetLastError() != cudaSuccess) goto fail;

        if (cudaDeviceSynchronize() != cudaSuccess) goto fail;
    }

    {
        ChessBoard *h_boards = (ChessBoard *)malloc(N * sizeof(ChessBoard));
        if (!h_boards) goto fail;
        if (cudaMemcpy(h_boards, d_boards, N * 64 * sizeof(int8_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            free(h_boards); goto fail;
        }
        cudaFree(d_states);
        cudaFree(d_boards);
        return h_boards;
    }

fail:
    fprintf(stderr, "[board_gen] CUDA error: %s\n",
            cudaGetErrorString(cudaGetLastError()));
    cudaFree(d_states);
    cudaFree(d_boards);
    return NULL;
}
