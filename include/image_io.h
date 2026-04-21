#pragma once
// image_io.h — PNG image I/O using libpng (no external stb dependency)

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// RGBA pixel (4 bytes per pixel, alpha always 255 for grayscale/RGB images)
typedef struct {
    uint8_t r, g, b, a;
} Pixel;

// Image structure
typedef struct {
    uint8_t *data;   // packed RGB or grayscale bytes
    int      width;
    int      height;
    int      channels; // 1 = gray, 3 = RGB
} Image;

// Load PNG from file. Returns 1 on success, 0 on failure.
// On success, img->data is malloc'd (caller must free).
// Loaded image is always converted to RGB (3 channels).
int image_load_png(const char *path, Image *img);

// Save grayscale (1-channel) image as PNG. Returns 1 on success.
int image_save_png_gray(const char *path, const uint8_t *data, int width, int height);

// Free image data
void image_free(Image *img);

// Collect all .png files in a directory.
// Returns malloc'd array of malloc'd strings. *count set to array length.
// Caller must free each string and the array.
char **collect_png_files(const char *dir, int *count);

// Save 8x8 float array as CSV (chess board intensity grid)
int save_intensity_csv(const char *path, const float *intensities);

#ifdef __cplusplus
}
#endif
