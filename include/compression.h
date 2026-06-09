#pragma once
// compression.h — GPU block-DCT image compression (JPEG-style)
//
// Stage 2 of the pipeline:
//   NPP  : RGB → grayscale conversion
//   CUDA : forward 8×8 DCT per block
//   CUDA : quantization (quality-factor scaled JPEG luminance table)
//   CUDA : dequantize + inverse DCT → reconstructed image
//   CUDA : parallel PSNR reduction

#include "image_io.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t *gray;    // reconstructed grayscale image (W*H bytes), malloc'd
    int      width;
    int      height;
    float    psnr;    // peak signal-to-noise ratio in dB (higher = better quality)
    float    ratio;   // fraction of non-zero quantized DCT coefficients (lower = more compression)
    int      quality; // quality factor used (1–100; 50 = standard JPEG midpoint)
} CompressedImage;

// Compress then reconstruct an RGB image using GPU block DCT.
//   img     : source RGB image loaded by image_load_png()
//   quality : 1–100  (50 = default; lower = more compression + more artifacts)
//   out     : filled by this function; call compression_free() when done
// Returns 1 on success, 0 on CUDA / NPP failure.
int compression_run(const Image *img, int quality, CompressedImage *out);

// Save the reconstructed grayscale image as a PNG file.
// Returns 1 on success, 0 on failure.
int compression_save_png(const char *path, const CompressedImage *c);

// Release the internal gray buffer.
void compression_free(CompressedImage *c);

#ifdef __cplusplus
}
#endif
