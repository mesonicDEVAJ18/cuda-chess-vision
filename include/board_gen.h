#pragma once
// board_gen.h — CUDA-driven chess position generation + pixel-art PNG renderer
//
// Stage 1 of the pipeline:
//   GPU: cuRAND kernels generate N random legal positions
//   CPU: pixel-art renderer writes each board to a 512×512 RGB PNG

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------
// Piece codes (stored in ChessBoard.squares[]):
//   0  = empty
//   1  = white pawn      -1 = black pawn
//   2  = white knight    -2 = black knight
//   3  = white bishop    -3 = black bishop
//   4  = white rook      -4 = black rook
//   5  = white queen     -5 = black queen
//   6  = white king      -6 = black king
//
// Square index: sq = rank * 8 + file
//   rank 0 = rank-1 (white's back rank, a1..h1)
//   rank 7 = rank-8 (black's back rank, a8..h8)
// -----------------------------------------------------------------------
typedef struct {
    int8_t squares[64];
} ChessBoard;

// Generate N random legal positions on the GPU using cuRAND.
//   seed  : RNG seed (vary per run for different boards)
// Returns malloc'd ChessBoard[N] on host; caller must free().
// Returns NULL on CUDA error.
ChessBoard *boards_generate_gpu(int N, unsigned long long seed);

// Render one ChessBoard to a color RGB PNG file.
//   path : output file path
//   b    : board to render
//   size : image side length in pixels (must be divisible by 8; use 512)
// Returns 1 on success, 0 on failure.
int board_save_png(const char *path, const ChessBoard *b, int size);

// Convert a ChessBoard to a float[64] vector for GPU evaluation.
// Piece values used:
//   pawn=1.0  knight=3.0  bishop=3.3  rook=5.0  queen=9.0  king=20.0
// White pieces → positive, black pieces → negative, empty → 0.0
void board_to_floats(const ChessBoard *b, float out[64]);

#ifdef __cplusplus
}
#endif
