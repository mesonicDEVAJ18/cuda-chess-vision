#pragma once
// pipeline.h — NPP + custom-kernel image processing pipeline

#include "image_io.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t *gray;           // equalized edge map (w*h bytes)
    uint8_t *edges;          // raw gradient magnitude (w*h bytes)
    float    intensities[64];// per-square mean intensity (8x8)
    int      width, height;
} PipelineResult;

int  pipeline_run        (const Image *img, PipelineResult *result);
void pipeline_result_free(PipelineResult *result);

#ifdef __cplusplus
}
#endif
