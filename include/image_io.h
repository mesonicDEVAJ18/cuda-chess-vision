#pragma once
// image_io.h — PNG I/O via libpng + directory scanning + CSV utilities

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// RGB image (3 bytes/pixel, packed, no alpha)
typedef struct {
    uint8_t *data;     // packed RGB: data[(y*width+x)*3 + channel]
    int      width;
    int      height;
    int      channels; // always 3 after load
} Image;

// Load a PNG file into an RGB Image struct.
// Returns 1 on success, 0 on failure. Caller must call image_free().
int image_load_png(const char *path, Image *img);

// Save a grayscale (1-channel) PNG.
int image_save_png_gray(const char *path, const uint8_t *data, int w, int h);

// Save an RGB (3-channel) PNG.
int image_save_png_rgb(const char *path, const uint8_t *data, int w, int h);

// Free the pixel buffer inside an Image struct.
void image_free(Image *img);

// Scan a directory and return all .png file paths (case-insensitive).
// count is filled with the number of files found.
// Caller must free each path string and the array itself.
char **collect_png_files(const char *dir, int *count);

// Write a per-square intensity table as an 8×8 CSV (files a-h, ranks 1-8).
int save_intensity_csv(const char *path, const float *intensities);

// Append one row to a compression stats CSV.
// Creates the file (with header) if it doesn't exist.
int compression_csv_append(const char *csv_path, const char *board_name,
                           int quality, float psnr, float ratio);

#ifdef __cplusplus
}
#endif
