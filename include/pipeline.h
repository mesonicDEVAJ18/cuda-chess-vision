#pragma once
// pipeline.h — GPU image processing pipeline interface

#include "image_io.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Output of a single image's GPU processing pass
typedef struct {
    uint8_t *gray;          // grayscale processed image (width * height bytes)
    uint8_t *edges;         // Sobel edge magnitude map (width * height bytes)
    float    intensities[64]; // 8x8 per-square mean intensities
    int      width;
    int      height;
} PipelineResult;

// Run the full GPU pipeline on one RGB image.
// Returns 1 on success, 0 on failure.
// On success, result fields are malloc'd — call pipeline_result_free().
int pipeline_run(const Image *img, PipelineResult *result);

// Free pipeline result buffers
void pipeline_result_free(PipelineResult *result);

#ifdef __cplusplus
}
#endif
