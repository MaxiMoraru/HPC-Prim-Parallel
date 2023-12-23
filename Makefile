# Directories
SRC_DIR := Source
INC_DIR := Headers
DATA_DIR := Data
BUILD_DIR := Build

# Compiler and flags
CC := mpicc
CFLAGS := -Wall -Wextra -I$(INC_DIR) -fopenmp

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c)

# Object files
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

# Targets
all: $(BUILD_DIR)/prim-mpi-open-mp $(BUILD_DIR)/prim

$(BUILD_DIR)/prim-mpi-open-mp: $(BUILD_DIR)/prim-mpi-open-mp.o
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD_DIR)/prim: $(BUILD_DIR)/prim.o
	$(CC) $(CFLAGS) $^ -o $@

$(BUILD_DIR)/prim-mpi-open-mp.o: $(SRC_DIR)/prim-mpi-open-mp.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/prim.o: $(SRC_DIR)/prim.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(BUILD_DIR)/*

test-mpi-open-mp: $(BUILD_DIR)/prim-mpi-open-mp
	@export OMP_NUM_THREADS=16;
	mpirun -np 8 ./$(BUILD_DIR)/prim-mpi-open-mp

test-prim: $(BUILD_DIR)/prim
	./$(BUILD_DIR)/prim
