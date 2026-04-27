#pragma once
// evaluator.h — GPU batch board evaluation (cuBLAS) + sorting (Thrust) + FFT (cuFFT)

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One board's evaluation result
typedef struct {
    float score;          // evaluation score (higher = better for white)
    float material;       // raw material balance
    float pawn_structure; // pawn structure FFT energy score
    int   rank;           // Thrust-sorted rank among all boards (0 = best)
    char  filename[256];  // source filename
} BoardEval;

// Evaluate N boards in batch.
// intensities : float[N][64] — per-square intensities from pipeline
// filenames   : const char*[N]
// results     : BoardEval[N] — output (allocated by caller)
int evaluator_run(
    const float  *intensities,  // N*64 floats, row-major
    const char  **filenames,
    int           N,
    BoardEval    *results);

// Save evaluation results to CSV
int evaluator_save_csv(const char *path, const BoardEval *results, int N);

#ifdef __cplusplus
}
#endif
