#pragma once
// image_io.h — PNG I/O via libpng + CSV/directory utilities

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t *data;    // packed RGB (3 bytes/pixel)
    int      width;
    int      height;
    int      channels; // always 3 after load
} Image;

int    image_load_png      (const char *path, Image *img);
int    image_save_png_gray (const char *path, const uint8_t *data, int w, int h);
void   image_free          (Image *img);
char **collect_png_files   (const char *dir, int *count);
int    save_intensity_csv  (const char *path, const float *intensities);

#ifdef __cplusplus
}
#endif
