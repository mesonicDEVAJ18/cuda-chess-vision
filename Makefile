# Makefile — CUDA Chess Vision
#
# Usage:
#   make            build release binary
#   make debug      build with debug symbols
#   make run        build and run on sample data
#   make clean      remove build artifacts

# ---------------------------------------------------------------------------
# Auto-detect CUDA path
# ---------------------------------------------------------------------------
ifndef CUDA_PATH
  NVCC_BIN := $(shell which nvcc 2>/dev/null)
  ifneq ($(NVCC_BIN),)
    CUDA_PATH := $(abspath $(dir $(NVCC_BIN))..)
  else
    CUDA_PATH := $(firstword $(wildcard \
      /usr/local/cuda \
      /usr/local/cuda-12.9 \
      /usr/local/cuda-12.8 \
      /usr/local/cuda-12.6 \
      /usr/local/cuda-12.4 \
      /usr/local/cuda-12.2 \
      /usr/local/cuda-12.0 \
      /usr/local/cuda-11.8 \
      /usr/local/cuda-11.7 \
      /usr/local/cuda-11.6 \
      /usr/cuda \
      /opt/cuda))
  endif
endif

ifeq ($(CUDA_PATH),)
  $(error Cannot find CUDA. Run: make CUDA_PATH=/usr/local/cuda-XX.X)
endif

$(info CUDA path: $(CUDA_PATH))

NVCC = $(CUDA_PATH)/bin/nvcc

# ---------------------------------------------------------------------------
# Host compiler — prefer gcc-12 (max supported by most nvcc), fall back to gcc
# ---------------------------------------------------------------------------
ifndef HOSTCC
  HOSTCC := $(firstword $(wildcard \
    /usr/bin/gcc-12 \
    /usr/local/bin/gcc-12) \
    $(shell which gcc-12 2>/dev/null))
endif

ifeq ($(HOSTCC),)
  HOSTCC      := $(firstword $(wildcard /usr/bin/gcc /usr/local/bin/gcc) \
                   $(shell which gcc 2>/dev/null))
  UNSUPPORTED := -allow-unsupported-compiler
  $(info gcc-12 not found, using $(HOSTCC) with -allow-unsupported-compiler)
else
  UNSUPPORTED :=
  $(info Host compiler: $(HOSTCC))
endif

ifeq ($(HOSTCC),)
  $(error Cannot find gcc. Run: sudo apt-get install build-essential)
endif

# ---------------------------------------------------------------------------
# NPP libraries
# CUDA 12.x split nppi into sub-libraries. Detect which ones exist.
# ---------------------------------------------------------------------------
CUDA_LIBDIR := $(CUDA_PATH)/lib64

# nppi (older monolithic) vs split libs (newer)
ifneq ($(wildcard $(CUDA_LIBDIR)/libnppi.so),)
  NPP_LIBS := -lnppc -lnppi
else
  # CUDA 12+ split layout — include all image-processing sub-libs
  NPP_LIBS := -lnppc -lnppig -lnppicc -lnppidei -lnppif -lnppim -lnppist -lnppisu -lnppitc -lnpps
endif

$(info NPP libs: $(NPP_LIBS))

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
GENCODE = \
  -gencode arch=compute_60,code=sm_60 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86

NVCCFLAGS = -std=c++14 -O2 $(GENCODE) -ccbin $(HOSTCC) $(UNSUPPORTED) -Xcompiler -Wall
CFLAGS    = -std=c11 -O2 -Wall -D_POSIX_C_SOURCE=200809L

INCLUDES  = -I include -I $(CUDA_PATH)/include

LDFLAGS   = -L $(CUDA_LIBDIR) $(NPP_LIBS) -lcudart -lpng -lm

# ---------------------------------------------------------------------------
# Sources and targets
# ---------------------------------------------------------------------------
BUILD_DIR = build
BIN       = chess_vision

CU_SRCS   = src/main.cu src/pipeline.cu
C_SRCS    = src/image_io.c

CU_OBJS   = $(CU_SRCS:src/%.cu=$(BUILD_DIR)/%.o)
C_OBJS    = $(C_SRCS:src/%.c=$(BUILD_DIR)/%.o)
ALL_OBJS  = $(CU_OBJS) $(C_OBJS)

.PHONY: all debug run clean

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
	@echo "  Build successful: ./$(BIN)"
	@echo "  Quick run: make run"
	@echo "  Full help: ./$(BIN) --help"
	@echo ""

run: all
	mkdir -p results
	./$(BIN) --input data/sample_boards --output results --csv --verbose

clean:
	rm -rf $(BUILD_DIR) $(BIN) results