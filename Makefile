# Makefile — CUDA Chess Vision (Redesigned)
#
# Three-stage pipeline:
#   Stage 1 : board_gen.cu    — cuRAND position generation
#   Stage 2 : compression.cu  — NPP + block-DCT compression
#   Stage 3 : evaluator.cu    — cuBLAS + cuFFT + Thrust evaluation
#
# Usage:
#   make              build
#   make run          build + run (20 boards, Q=50)
#   make run-large    build + run (100 boards)
#   make viz          run visualise.py on last results
#   make clean        remove build artefacts

# ---------------------------------------------------------------------------
# CUDA / nvcc detection
# Two layouts supported:
#   A) System install (Ubuntu 22.04 nvidia-cuda-toolkit):
#      nvcc at /usr/bin/nvcc, headers in /usr/include, libs in /usr/lib/...
#   B) Upstream installer: nvcc at /usr/local/cuda/bin/nvcc
# ---------------------------------------------------------------------------
NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
  NVCC := $(firstword $(wildcard \
    /usr/local/cuda/bin/nvcc \
    /usr/local/cuda-12.0/bin/nvcc \
    /usr/local/cuda-11.8/bin/nvcc))
endif
ifeq ($(NVCC),)
  $(error Cannot find nvcc. Install CUDA toolkit first.)
endif
$(info nvcc     : $(NVCC))

# Derive CUDA_PATH from nvcc location (works for both layouts)
NVCC_DIR  := $(dir $(NVCC))
CUDA_PATH := $(abspath $(NVCC_DIR)/..)
$(info CUDA_PATH: $(CUDA_PATH))

# Library directory — system layout puts libs in /usr/lib/x86_64-linux-gnu
# upstream puts them in $(CUDA_PATH)/lib64
CUDA_LIBDIR := $(CUDA_PATH)/lib64
ifeq ($(wildcard $(CUDA_LIBDIR)/libcudart.so),)
  CUDA_LIBDIR := /usr/lib/x86_64-linux-gnu
endif
$(info lib dir  : $(CUDA_LIBDIR))

# Include directory — system layout uses /usr/include
CUDA_INC := $(CUDA_PATH)/include
ifeq ($(wildcard $(CUDA_INC)/cuda_runtime.h),)
  CUDA_INC := /usr/include
endif
$(info inc dir  : $(CUDA_INC))

# ---------------------------------------------------------------------------
# Host compiler
# ---------------------------------------------------------------------------
ifndef HOSTCC
  HOSTCC := $(firstword \
    $(wildcard /usr/bin/gcc-12 /usr/local/bin/gcc-12) \
    $(shell which gcc-12 2>/dev/null))
endif
ifeq ($(HOSTCC),)
  HOSTCC      := $(firstword $(wildcard /usr/bin/gcc /usr/local/bin/gcc) \
                   $(shell which gcc 2>/dev/null))
  UNSUPPORTED := -allow-unsupported-compiler
  $(info Host CC  : $(HOSTCC) [with -allow-unsupported-compiler])
else
  UNSUPPORTED :=
  $(info Host CC  : $(HOSTCC))
endif
ifeq ($(HOSTCC),)
  $(error Cannot find gcc.)
endif

# ---------------------------------------------------------------------------
# NPP library detection (monolithic vs split vs system path)
# ---------------------------------------------------------------------------
ifneq ($(wildcard $(CUDA_LIBDIR)/libnppi.so),)
  NPP_LIBS := -lnppc -lnppi
else
  # System or split layout — include all component libraries
  NPP_LIBS := -lnppc -lnppig -lnppicc -lnppidei -lnppif \
              -lnppim -lnppist -lnppisu -lnppitc -lnpps -lnppial
endif
$(info NPP libs : $(NPP_LIBS))

# ---------------------------------------------------------------------------
# GPU architecture targets
# ---------------------------------------------------------------------------
GENCODE = \
  -gencode arch=compute_60,code=sm_60 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
NVCCFLAGS = -std=c++14 -O2 $(GENCODE) -ccbin $(HOSTCC) $(UNSUPPORTED) \
            -Xcompiler -Wall
CFLAGS    = -std=c11 -O2 -Wall -D_POSIX_C_SOURCE=200809L
INCLUDES  = -I include -I $(CUDA_INC)

# System layout: libs are already on the linker's default search path,
# but we add $(CUDA_LIBDIR) explicitly for good measure.
# -lstdc++ is required for Thrust.
LDFLAGS = \
  -L $(CUDA_LIBDIR) \
  $(NPP_LIBS) \
  -lcublas_static -lcublasLt_static -lculibos \
  -lcufft \
  -lcudart \
  -lpng -lm -lstdc++

# ---------------------------------------------------------------------------
# Sources & objects
# ---------------------------------------------------------------------------
BUILD_DIR = build
BIN       = chess_vision

CU_SRCS = src/main.cu \
          src/board_gen.cu \
          src/compression.cu \
          src/evaluator.cu

C_SRCS  = src/image_io.c

CU_OBJS = $(CU_SRCS:src/%.cu=$(BUILD_DIR)/%.o)
C_OBJS  = $(C_SRCS:src/%.c=$(BUILD_DIR)/%.o)
ALL_OBJS = $(CU_OBJS) $(C_OBJS)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all debug run run-large viz clean

all: $(BUILD_DIR) $(BIN)

debug: NVCCFLAGS += -G -g
debug: CFLAGS    += -g
debug: all

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.c
	$(HOSTCC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BIN): $(ALL_OBJS)
	$(NVCC) $(GENCODE) -ccbin $(HOSTCC) $(UNSUPPORTED) $^ -o $@ $(LDFLAGS)
	@echo ""
	@echo "  ✓  Build successful: ./$(BIN)"
	@echo "  →  Quick run: make run"
	@echo ""

run: all
	mkdir -p results/boards results/compressed
	./$(BIN) --boards 20 --quality 50 --output results --verbose

run-large: all
	mkdir -p results/boards results/compressed
	./$(BIN) --boards 100 --quality 50 --output results --verbose

viz:
	python3 scripts/visualize.py --results results --output plots

clean:
	rm -rf $(BUILD_DIR) $(BIN) results plots