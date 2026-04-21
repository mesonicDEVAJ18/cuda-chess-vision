# Makefile — CUDA Chess Vision
# Tested on Coursera CUDA lab (Ubuntu, CUDA 11/12, NVIDIA T4)
#
# Usage:
#   make          — build release binary
#   make debug    — build with debug symbols
#   make run      — build and run on sample data
#   make clean    — remove build artifacts

CUDA_PATH  ?= /usr/local/cuda

NVCC        = $(CUDA_PATH)/bin/nvcc
CC          = gcc

# Generate PTX for a wide range of GPU architectures (T4 = sm_75)
GENCODE     = -gencode arch=compute_60,code=sm_60 \
              -gencode arch=compute_70,code=sm_70 \
              -gencode arch=compute_75,code=sm_75 \
              -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_86,code=sm_86

NVCCFLAGS   = -std=c++14 -O2 $(GENCODE) -Xcompiler -Wall
CFLAGS      = -std=c11 -O2 -Wall

INCLUDES    = -I include -I $(CUDA_PATH)/include

# NPP libs needed: core (nppc), image processing (nppi)
# libpng for PNG I/O
LDFLAGS     = -L $(CUDA_PATH)/lib64 \
              -lnppc -lnppi \
              -lcudart \
              -lpng -lm

BUILD_DIR   = build
BIN         = chess_vision

# Source files
CU_SRCS     = src/main.cu src/pipeline.cu
C_SRCS      = src/image_io.c

CU_OBJS     = $(CU_SRCS:src/%.cu=$(BUILD_DIR)/%.o)
C_OBJS      = $(C_SRCS:src/%.c=$(BUILD_DIR)/%.o)
ALL_OBJS    = $(CU_OBJS) $(C_OBJS)

.PHONY: all debug run clean

all: $(BUILD_DIR) $(BIN)

debug: NVCCFLAGS += -G -g
debug: CFLAGS    += -g
debug: all

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile .cu files
$(BUILD_DIR)/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Compile .c files (via nvcc so linking is consistent)
$(BUILD_DIR)/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Link
$(BIN): $(ALL_OBJS)
	$(NVCC) $(GENCODE) $^ -o $@ $(LDFLAGS)
	@echo ""
	@echo "  Build successful: ./$(BIN)"
	@echo "  Run:  make run"
	@echo "  Help: ./$(BIN) --help"
	@echo ""

run: all
	mkdir -p results
	./$(BIN) --input data/sample_boards --output results --csv --verbose

clean:
	rm -rf $(BUILD_DIR) $(BIN) results
