// image_io.c — PNG image I/O using libpng, CSV export, directory scanning

#define _POSIX_C_SOURCE 200809L   // enables strdup, readdir, etc.

#include "image_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <strings.h>
#include <png.h>

int image_load_png(const char *path, Image *img) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[image_io] Cannot open: %s\n", path);
        return 0;
    }

    uint8_t sig[8];
    if (fread(sig, 1, 8, fp) != 8 || png_sig_cmp(sig, 0, 8)) {
        fprintf(stderr, "[image_io] Not a valid PNG: %s\n", path);
        fclose(fp);
        return 0;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return 0; }

    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, NULL, NULL); fclose(fp); return 0; }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(fp);
        return 0;
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    int width      = (int)png_get_image_width(png, info);
    int height     = (int)png_get_image_height(png, info);
    int color_type = png_get_color_type(png, info);
    int bit_depth  = png_get_bit_depth(png, info);

    if (bit_depth == 16)                          png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE)     png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
                                                  png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS))  png_set_tRNS_to_alpha(png);
    if (color_type & PNG_COLOR_MASK_ALPHA)        png_set_strip_alpha(png);
    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)  png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    size_t row_bytes = png_get_rowbytes(png, info);
    uint8_t *data = (uint8_t *)malloc(height * row_bytes);
    if (!data) { png_destroy_read_struct(&png, &info, NULL); fclose(fp); return 0; }

    png_bytep *rows = (png_bytep *)malloc(height * sizeof(png_bytep));
    if (!rows) { free(data); png_destroy_read_struct(&png, &info, NULL); fclose(fp); return 0; }
    for (int y = 0; y < height; y++)
        rows[y] = data + y * row_bytes;

    png_read_image(png, rows);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);
    free(rows);

    img->data     = data;
    img->width    = width;
    img->height   = height;
    img->channels = 3;
    return 1;
}

int image_save_png_gray(const char *path, const uint8_t *data, int width, int height) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[image_io] Cannot write: %s\n", path);
        return 0;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return 0; }

    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, NULL); fclose(fp); return 0; }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return 0;
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8,
                 PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    for (int y = 0; y < height; y++)
        png_write_row(png, (png_bytep)(data + y * width));

    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return 1;
}

void image_free(Image *img) {
    if (img && img->data) {
        free(img->data);
        img->data = NULL;
    }
}

static int is_png(const char *name) {
    const char *dot = strrchr(name, '.');
    return dot && strcasecmp(dot, ".png") == 0;
}

char **collect_png_files(const char *dir, int *count) {
    DIR *d = opendir(dir);
    if (!d) {
        fprintf(stderr, "[image_io] Cannot open dir: %s\n", dir);
        *count = 0;
        return NULL;
    }

    struct dirent *entry;
    int n = 0;
    while ((entry = readdir(d)) != NULL)
        if (is_png(entry->d_name)) n++;
    rewinddir(d);

    char **paths = (char **)malloc(n * sizeof(char *));
    if (!paths) { closedir(d); *count = 0; return NULL; }

    int i = 0;
    while ((entry = readdir(d)) != NULL && i < n) {
        if (is_png(entry->d_name)) {
            char buf[2048];
            snprintf(buf, sizeof(buf), "%s/%s", dir, entry->d_name);
            paths[i++] = strdup(buf);
        }
    }
    closedir(d);
    *count = i;
    return paths;
}

int save_intensity_csv(const char *path, const float *intensities) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fputs("rank,a,b,c,d,e,f,g,h\n", f);
    for (int r = 7; r >= 0; r--) {
        fprintf(f, "%d", r + 1);
        for (int c = 0; c < 8; c++)
            fprintf(f, ",%.2f", intensities[r * 8 + c]);
        fputc('\n', f);
    }
    fclose(f);
    return 1;
}