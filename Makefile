# Directories
SRC_DIR := Source
INC_DIR := Headers
DATA_DIR := Data
BUILD_DIR := Build

# Compilers
GCC := gcc
MPI_CC := mpicc
CUDA_NVCC := nvcc

# Compiler flags
CFLAGS := -Wall -Wextra -I$(INC_DIR)
CFLAGS_OPENMP := $(CFLAGS) -fopenmp
CUDA_FLAGS := -Xcompiler -fopenmp


# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c)

# Object files
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

# Targets
all: $(BUILD_DIR)/prim-mpi-open-mp $(BUILD_DIR)/prim-mpi $(BUILD_DIR)/prim $(BUILD_DIR)/prim-cuda-open-mp

$(BUILD_DIR)/prim-mpi-open-mp: $(BUILD_DIR)/prim-mpi-open-mp.o
	$(MPI_CC) $(CFLAGS_OPENMP) $^ -o $@

$(BUILD_DIR)/prim-mpi: $(BUILD_DIR)/prim-mpi.o
	$(MPI_CC) $(CFLAGS) $^ -o $@

$(BUILD_DIR)/prim: $(BUILD_DIR)/prim.o
	$(GCC) $(CFLAGS) $^ -o $@

$(BUILD_DIR)/prim-cuda-open-mp: $(BUILD_DIR)/prim-cuda-open-mp.o
	$(CUDA_NVCC) $(CUDA_FLAGS) $^ -o $@

$(BUILD_DIR)/prim-mpi-open-mp.o: $(SRC_DIR)/prim-mpi-open-mp.c
	$(MPI_CC) $(CFLAGS_OPENMP) -c $< -o $@

$(BUILD_DIR)/prim-mpi.o: $(SRC_DIR)/prim-mpi-open-mp.c
	$(MPI_CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/prim.o: $(SRC_DIR)/prim.c
	$(GCC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/prim-cuda-open-mp.o: $(SRC_DIR)/prim-cuda-open-mp.cu
	$(CUDA_NVCC) $(CUDA_FLAGS) -c $< -o $@

clean:
	rm -f $(BUILD_DIR)/*

test-mpi-open-mp: $(BUILD_DIR)/prim-mpi-open-mp
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR)/prim-mpi-open-mp

test-mpi: $(BUILD_DIR)/prim-mpi
	mpirun -np 8 ./$(BUILD_DIR)/prim-mpi

test-prim: $(BUILD_DIR)/prim
	./$(BUILD_DIR)/prim

test-cuda-open-mp: $(BUILD_DIR)/prim-cuda-open-mp
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	./$(BUILD_DIR)/prim-cuda-open-mp