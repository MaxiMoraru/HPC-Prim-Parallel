#-Moraru, Maximilian Marius, MAT. 0622702167, m.moraru@studenti.unisa.it
#-Algoritmo “Prim”
#-Moscato Francesco, fmoscato@unisa.it
#-Final Projects are assigned to each student. Student shall provide a parallel version of an algorithm with both "OpenMP + MPI" and "OpenMP + Cuda" approaches, comparing results with a known solution on single-processing node. Results and differences shall be discussed for different inputs (type and size). The parallel algorithm used in "OpenMP + MPI" solution could not be the same of the "OpenMP + CUDA" approach.


#    prim.c
#   Prim's algorithm for finding the minimum spanning tree of a graph

    
#Copyright (C) <2024>  <Maximilian Marius Moraru>

#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.




# Directories
SRC_DIR := Source
INC_DIR := Headers
DATA_DIR := Data
BUILD_DIR := Build

BUILD_DIR_0 := $(BUILD_DIR)/O0
BUILD_DIR_1 := $(BUILD_DIR)/O1
BUILD_DIR_2 := $(BUILD_DIR)/O2
BUILD_DIR_3 := $(BUILD_DIR)/O3
BUILD_DIR_PRIM := $(BUILD_DIR)/Prim

# Compilers
GCC := gcc
MPI_CC := mpicc
CUDA_NVCC := nvcc



# Compiler flags -O0
CFLAGS_OPENMP_0 := $(CFLAGS) -fopenmp
CUDA_FLAGS_0 := -Xcompiler "-O0 -Wall -Wextra -I$(INC_DIR)"
CUDA_FLAGS_OPENMP_0 := -Xcompiler "-O0 -Wall -Wextra -I$(INC_DIR) -fopenmp"

# Compiler flags -O1
CFLAGS_OPENMP_1 := $(CFLAGS) -fopenmp
CUDA_FLAGS_1 := -Xcompiler "-O1 -Wall -Wextra -I$(INC_DIR)"
CUDA_FLAGS_OPENMP_1:= -Xcompiler "-O1 -Wall -Wextra -I$(INC_DIR) -fopenmp"

# Compiler flags -O2
CFLAGS_OPENMP_2 := $(CFLAGS) -fopenmp
CUDA_FLAGS_2 := -Xcompiler "-O2 -Wall -Wextra -I$(INC_DIR)"
CUDA_FLAGS_OPENMP_2 := -Xcompiler "-O2 -Wall -Wextra -I$(INC_DIR) -fopenmp"

# Compiler flags -O3
CFLAGS_OPENMP_3 := $(CFLAGS) -fopenmp
CUDA_FLAGS_3 := -Xcompiler "-O3 -Wall -Wextra -I$(INC_DIR)"
CUDA_FLAGS_OPENMP_3 := -Xcompiler "-O3 -Wall -Wextra -I$(INC_DIR) -fopenmp"

# Compiler flags -Prim
PRIM_FLAGS :=  -Wall -Wextra -I$(INC_DIR)



# Source files
SRCS := $(wildcard $(SRC_DIR)/*.c)

# Object files
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

# Targets
all: programs_0 programs_1 programs_2 programs_3 program_prim

programs_0: \
	$(BUILD_DIR_0)/prim-mpi-open-mp-0 \
	$(BUILD_DIR_0)/prim-mpi-0 \
	$(BUILD_DIR_0)/prim-cuda-open-mp-0 \
	$(BUILD_DIR_0)/prim-cuda-0 

programs_1: \
	$(BUILD_DIR_1)/prim-mpi-open-mp-1 \
	$(BUILD_DIR_1)/prim-mpi-1 \
	$(BUILD_DIR_1)/prim-cuda-open-mp-1 \
	$(BUILD_DIR_1)/prim-cuda-1 

programs_2: \
	$(BUILD_DIR_2)/prim-mpi-open-mp-2 \
	$(BUILD_DIR_2)/prim-mpi-2 \
	$(BUILD_DIR_2)/prim-cuda-open-mp-2 \
	$(BUILD_DIR_2)/prim-cuda-2 

programs_3: \
	$(BUILD_DIR_3)/prim-mpi-open-mp-3 \
	$(BUILD_DIR_3)/prim-mpi-3 \
	$(BUILD_DIR_3)/prim-cuda-open-mp-3 \
	$(BUILD_DIR_3)/prim-cuda-3 

program_prim: \
	$(BUILD_DIR_PRIM)/prim 
#############################################
# -O0
#############################################
# mpi-open-mp
$(BUILD_DIR_0)/prim-mpi-open-mp-0: $(BUILD_DIR_0)/prim-mpi-open-mp-0.o
	$(MPI_CC) $(CFLAGS_OPENMP_0) $^ -o $@
# mpi
$(BUILD_DIR_0)/prim-mpi-0: $(BUILD_DIR_0)/prim-mpi-0.o
	$(MPI_CC) $(CFLAGS_0) $^ -o $@
# cuda-open-mp
$(BUILD_DIR_0)/prim-cuda-open-mp-0: $(BUILD_DIR_0)/prim-cuda-open-mp-0.o
	$(CUDA_NVCC) $(CUDA_FLAGS_OPENMP_0) $^ -o $@
# cuda
$(BUILD_DIR_0)/prim-cuda-0: $(BUILD_DIR_0)/prim-cuda-0.o
	$(CUDA_NVCC) $(CUDA_FLAGS_0) $^ -o $@

#############################################

# mpi-open-mp
$(BUILD_DIR_0)/prim-mpi-open-mp-0.o: $(SRC_DIR)/prim-mpi-open-mp.c
	$(MPI_CC) $(CFLAGS_OPENMP_0) -c $< -o $@
# mpi
$(BUILD_DIR_0)/prim-mpi-0.o: $(SRC_DIR)/prim-mpi-open-mp.c
	$(MPI_CC) $(CFLAGS_0) -c $< -o $@
# cuda-open-mp
$(BUILD_DIR_0)/prim-cuda-open-mp-0.o: $(SRC_DIR)/prim-cuda-open-mp.cu
	$(CUDA_NVCC) $(CUDA_FLAGS_OPENMP_0) -c $< -o $@
# cuda
$(BUILD_DIR_0)/prim-cuda-0.o: $(SRC_DIR)/prim-cuda-open-mp.cu
	$(CUDA_NVCC) $(CUDA_FLAGS_0) -c $< -o $@

#############################################
# -O1
#############################################
# mpi-open-mp
$(BUILD_DIR_1)/prim-mpi-open-mp-1: $(BUILD_DIR_1)/prim-mpi-open-mp-1.o
	$(MPI_CC) $(CFLAGS_OPENMP_1) $^ -o $@
# mpi
$(BUILD_DIR_1)/prim-mpi-1: $(BUILD_DIR_1)/prim-mpi-1.o
	$(MPI_CC) $(CFLAGS_1) $^ -o $@
# cuda-open-mp
$(BUILD_DIR_1)/prim-cuda-open-mp-1: $(BUILD_DIR_1)/prim-cuda-open-mp-1.o
	$(CUDA_NVCC) $(CUDA_FLAGS_OPENMP_1) $^ -o $@
# cuda
$(BUILD_DIR_1)/prim-cuda-1: $(BUILD_DIR_1)/prim-cuda-1.o
	$(CUDA_NVCC) $(CUDA_FLAGS_1) $^ -o $@

#############################################

# mpi-open-mp
$(BUILD_DIR_1)/prim-mpi-open-mp-1.o: $(SRC_DIR)/prim-mpi-open-mp.c
	$(MPI_CC) $(CFLAGS_OPENMP_1) -c $< -o $@
# mpi
$(BUILD_DIR_1)/prim-mpi-1.o: $(SRC_DIR)/prim-mpi-open-mp.c
	$(MPI_CC) $(CFLAGS_1) -c $< -o $@
# cuda-open-mp
$(BUILD_DIR_1)/prim-cuda-open-mp-1.o: $(SRC_DIR)/prim-cuda-open-mp.cu
	$(CUDA_NVCC) $(CUDA_FLAGS_OPENMP_1) -c $< -o $@
# cuda
$(BUILD_DIR_1)/prim-cuda-1.o: $(SRC_DIR)/prim-cuda-open-mp.cu
	$(CUDA_NVCC) $(CUDA_FLAGS_1) -c $< -o $@


#############################################
# -O2
#############################################
# mpi-open-mp
$(BUILD_DIR_2)/prim-mpi-open-mp-2: $(BUILD_DIR_2)/prim-mpi-open-mp-2.o
	$(MPI_CC) $(CFLAGS_OPENMP_2) $^ -o $@
# mpi
$(BUILD_DIR_2)/prim-mpi-2: $(BUILD_DIR_2)/prim-mpi-2.o
	$(MPI_CC) $(CFLAGS_2) $^ -o $@
# cuda-open-mp
$(BUILD_DIR_2)/prim-cuda-open-mp-2: $(BUILD_DIR_2)/prim-cuda-open-mp-2.o
	$(CUDA_NVCC) $(CUDA_FLAGS_OPENMP_2) $^ -o $@
# cuda
$(BUILD_DIR_2)/prim-cuda-2: $(BUILD_DIR_2)/prim-cuda-2.o
	$(CUDA_NVCC) $(CUDA_FLAGS_2) $^ -o $@

#############################################

# mpi-open-mp
$(BUILD_DIR_2)/prim-mpi-open-mp-2.o: $(SRC_DIR)/prim-mpi-open-mp.c
	$(MPI_CC) $(CFLAGS_OPENMP_2) -c $< -o $@
# mpi
$(BUILD_DIR_2)/prim-mpi-2.o: $(SRC_DIR)/prim-mpi-open-mp.c
	$(MPI_CC) $(CFLAGS_2) -c $< -o $@
# cuda-open-mp
$(BUILD_DIR_2)/prim-cuda-open-mp-2.o: $(SRC_DIR)/prim-cuda-open-mp.cu
	$(CUDA_NVCC) $(CUDA_FLAGS_OPENMP_2) -c $< -o $@
# cuda
$(BUILD_DIR_2)/prim-cuda-2.o: $(SRC_DIR)/prim-cuda-open-mp.cu
	$(CUDA_NVCC) $(CUDA_FLAGS_2) -c $< -o $@

#############################################
# -O3
#############################################
# mpi-open-mp
$(BUILD_DIR_3)/prim-mpi-open-mp-3: $(BUILD_DIR_3)/prim-mpi-open-mp-3.o
	$(MPI_CC) $(CFLAGS_OPENMP_3) $^ -o $@
# mpi
$(BUILD_DIR_3)/prim-mpi-3: $(BUILD_DIR_3)/prim-mpi-3.o
	$(MPI_CC) $(CFLAGS_3) $^ -o $@
# cuda-open-mp
$(BUILD_DIR_3)/prim-cuda-open-mp-3: $(BUILD_DIR_3)/prim-cuda-open-mp-3.o
	$(CUDA_NVCC) $(CUDA_FLAGS_OPENMP_3) $^ -o $@
# cuda
$(BUILD_DIR_3)/prim-cuda-3: $(BUILD_DIR_3)/prim-cuda-3.o
	$(CUDA_NVCC) $(CUDA_FLAGS_3) $^ -o $@
# prim
$(BUILD_DIR_3)/prim-3: $(BUILD_DIR_3)/prim-3.o
	$(GCC) $(CFLAGS_3) $^ -o $@

#############################################

# mpi-open-mp
$(BUILD_DIR_3)/prim-mpi-open-mp-3.o: $(SRC_DIR)/prim-mpi-open-mp.c
	$(MPI_CC) $(CFLAGS_OPENMP_3) -c $< -o $@
# mpi
$(BUILD_DIR_3)/prim-mpi-3.o: $(SRC_DIR)/prim-mpi-open-mp.c
	$(MPI_CC) $(CFLAGS_3) -c $< -o $@
# cuda-open-mp
$(BUILD_DIR_3)/prim-cuda-open-mp-3.o: $(SRC_DIR)/prim-cuda-open-mp.cu
	$(CUDA_NVCC) $(CUDA_FLAGS_OPENMP_3) -c $< -o $@
# cuda
$(BUILD_DIR_3)/prim-cuda-3.o: $(SRC_DIR)/prim-cuda-open-mp.cu
	$(CUDA_NVCC) $(CUDA_FLAGS_3) -c $< -o $@
# prim
$(BUILD_DIR_3)/prim-3.o: $(SRC_DIR)/prim.c
	$(GCC) $(CFLAGS_3) -c $< -o $@


#############################################
# Prim

$(BUILD_DIR_PRIM)/prim: $(BUILD_DIR_PRIM)/prim.o
	$(GCC) $(PRIM_FLAGS) $^ -o $@

$(BUILD_DIR_PRIM)/prim.o: $(SRC_DIR)/prim.c
	$(GCC) $(PRIM_FLAGS) -c $< -o $@

#############################################
clean:
	rm -f $(BUILD_DIR_0)/*
	rm -f $(BUILD_DIR_1)/*
	rm -f $(BUILD_DIR_2)/*
	rm -f $(BUILD_DIR_3)/*
	rm -f $(BUILD_DIR_PRIM)/*


#############################################

# Test
test: test-O0 test-O1 test-O2 test-O3 test-prim


#############################################

# Test -O0
test-O0: test-mpi-O0 test-cuda-O0

#############

# Test mpi

#############
test-mpi-O0: test-mpi-0-p2 test-mpi-0-p4 test-mpi-0-p8
# Test 2 processes
test-mpi-0-p2: test-mpi-0-p2-t0 test-mpi-0-p2-t2 test-mpi-0-p2-t4 test-mpi-0-p2-t8 test-mpi-0-p2-t16
# test n threads
test-mpi-0-p2-t0: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	mpirun -np 2 ./$(BUILD_DIR_0)/prim-mpi-0 100

test-mpi-0-p2-t2: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_0)/prim-mpi-open-mp-0 100

test-mpi-0-p2-t4: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_0)/prim-mpi-open-mp-0 100

test-mpi-0-p2-t8: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_0)/prim-mpi-open-mp-0 100

test-mpi-0-p2-t16: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_0)/prim-mpi-open-mp-0 100

#############
# test 4 processes
test-mpi-0-p4: test-mpi-0-p4-t0 test-mpi-0-p4-t2 test-mpi-0-p4-t4 test-mpi-0-p4-t8 test-mpi-0-p4-t16
#test n threads
test-mpi-0-p4-t0: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	mpirun -np 4 ./$(BUILD_DIR_0)/prim-mpi-0 100

test-mpi-0-p4-t2: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_0)/prim-mpi-open-mp-0 100

test-mpi-0-p4-t4: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_0)/prim-mpi-open-mp-0 100

test-mpi-0-p4-t8: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_0)/prim-mpi-open-mp-0 100

test-mpi-0-p4-t16: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_0)/prim-mpi-open-mp-0 100

#############
#test 8 processes
test-mpi-0-p8: test-mpi-0-p8-t0 test-mpi-0-p8-t2 test-mpi-0-p8-t4 test-mpi-0-p8-t8 test-mpi-0-p8-t16
#test n threads
test-mpi-0-p8-t0: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	mpirun -np 8 ./$(BUILD_DIR_0)/prim-mpi-0 100

test-mpi-0-p8-t2: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_0)/prim-mpi-open-mp-0 100

test-mpi-0-p8-t4: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_0)/prim-mpi-open-mp-0 100

test-mpi-0-p8-t8: $(BUILD_DIR_0)/prim-mpi-open-mp-0
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_0)/prim-mpi-open-mp-0 100

test-mpi-0-p8-t16: $(BUILD_DIR_0)/prim-mpi-open-mp-0	
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_0)/prim-mpi-open-mp-0 100

#############

# Test cuda

#############
test-cuda-O0: test-cuda-0-t0 test-cuda-0-t2 test-cuda-0-t4 test-cuda-0-t8 test-cuda-0-t16
#test n threads
test-cuda-0-t0: $(BUILD_DIR_0)/prim-cuda-0
	./$(BUILD_DIR_0)/prim-cuda-0 100

test-cuda-0-t2: $(BUILD_DIR_0)/prim-cuda-open-mp-0
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	./$(BUILD_DIR_0)/prim-cuda-open-mp-0 100

test-cuda-0-t4: $(BUILD_DIR_0)/prim-cuda-open-mp-0
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	./$(BUILD_DIR_0)/prim-cuda-open-mp-0 100	

test-cuda-0-t8: $(BUILD_DIR_0)/prim-cuda-open-mp-0
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	./$(BUILD_DIR_0)/prim-cuda-open-mp-0 100

test-cuda-0-t16: $(BUILD_DIR_0)/prim-cuda-open-mp-0
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	./$(BUILD_DIR_0)/prim-cuda-open-mp-0 100

#############

# Test prim

#############


#####################################################################################
# Test -O1
test-O1: test-mpi-O1 test-cuda-O1 

#############

# Test mpi

#############
test-mpi-O1: test-mpi-1-p2 test-mpi-1-p4 test-mpi-1-p8
# Test 2 processes
test-mpi-1-p2: test-mpi-1-p2-t0 test-mpi-1-p2-t2 test-mpi-1-p2-t4 test-mpi-1-p2-t8 test-mpi-1-p2-t16
# test n threads
test-mpi-1-p2-t0: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	mpirun -np 2 ./$(BUILD_DIR_1)/prim-mpi-1 100

test-mpi-1-p2-t2: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_1)/prim-mpi-open-mp-1 100

test-mpi-1-p2-t4: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_1)/prim-mpi-open-mp-1 100

test-mpi-1-p2-t8: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_1)/prim-mpi-open-mp-1 100

test-mpi-1-p2-t16: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_1)/prim-mpi-open-mp-1 100

#############
# test 4 processes
test-mpi-1-p4: test-mpi-1-p4-t0 test-mpi-1-p4-t2 test-mpi-1-p4-t4 test-mpi-1-p4-t8 test-mpi-1-p4-t16
#test n threads
test-mpi-1-p4-t0: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	mpirun -np 4 ./$(BUILD_DIR_1)/prim-mpi-1 100

test-mpi-1-p4-t2: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_1)/prim-mpi-open-mp-1 100

test-mpi-1-p4-t4: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_1)/prim-mpi-open-mp-1 100

test-mpi-1-p4-t8: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_1)/prim-mpi-open-mp-1 100

test-mpi-1-p4-t16: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_1)/prim-mpi-open-mp-1 100

#############
#test 8 processes
test-mpi-1-p8: test-mpi-1-p8-t0 test-mpi-1-p8-t2 test-mpi-1-p8-t4 test-mpi-1-p8-t8 test-mpi-1-p8-t16
#test n threads
test-mpi-1-p8-t0: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	mpirun -np 8 ./$(BUILD_DIR_1)/prim-mpi-1 100
	
test-mpi-1-p8-t2: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_1)/prim-mpi-open-mp-1 100

test-mpi-1-p8-t4: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_1)/prim-mpi-open-mp-1 100

test-mpi-1-p8-t8: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_1)/prim-mpi-open-mp-1 100

test-mpi-1-p8-t16: $(BUILD_DIR_1)/prim-mpi-open-mp-1
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_1)/prim-mpi-open-mp-1 100

#############

# Test cuda

#############
test-cuda-O1: test-cuda-1-t0 test-cuda-1-t2 test-cuda-1-t4 test-cuda-1-t8 test-cuda-1-t16
#test n threads
test-cuda-1-t0: $(BUILD_DIR_1)/prim-cuda-1
	./$(BUILD_DIR_1)/prim-cuda-1 100

test-cuda-1-t2: $(BUILD_DIR_1)/prim-cuda-open-mp-1
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	./$(BUILD_DIR_1)/prim-cuda-open-mp-1 100

test-cuda-1-t4: $(BUILD_DIR_1)/prim-cuda-open-mp-1
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	./$(BUILD_DIR_1)/prim-cuda-open-mp-1 100

test-cuda-1-t8: $(BUILD_DIR_1)/prim-cuda-open-mp-1
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	./$(BUILD_DIR_1)/prim-cuda-open-mp-1 100

test-cuda-1-t16: $(BUILD_DIR_1)/prim-cuda-open-mp-1
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	./$(BUILD_DIR_1)/prim-cuda-open-mp-1 100

#############

# Test prim

#############

#####################################################################################
# Test -O2
test-O2: test-mpi-O2 test-cuda-O2 

#############

# Test mpi

#############
test-mpi-O2: test-mpi-2-p2 test-mpi-2-p4 test-mpi-2-p8
# Test 2 processes
test-mpi-2-p2: test-mpi-2-p2-t0 test-mpi-2-p2-t2 test-mpi-2-p2-t4 test-mpi-2-p2-t8 test-mpi-2-p2-t16
# test n threads
test-mpi-2-p2-t0: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	mpirun -np 2 ./$(BUILD_DIR_2)/prim-mpi-2 100

test-mpi-2-p2-t2: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_2)/prim-mpi-open-mp-2 100

test-mpi-2-p2-t4: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_2)/prim-mpi-open-mp-2 100

test-mpi-2-p2-t8: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_2)/prim-mpi-open-mp-2 100

test-mpi-2-p2-t16: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_2)/prim-mpi-open-mp-2 100

#############
# test 4 processes
test-mpi-2-p4: test-mpi-2-p4-t0 test-mpi-2-p4-t2 test-mpi-2-p4-t4 test-mpi-2-p4-t8 test-mpi-2-p4-t16
#test n threads
test-mpi-2-p4-t0: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	mpirun -np 4 ./$(BUILD_DIR_2)/prim-mpi-2 100

test-mpi-2-p4-t2: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_2)/prim-mpi-open-mp-2 100

test-mpi-2-p4-t4: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_2)/prim-mpi-open-mp-2 100

test-mpi-2-p4-t8: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_2)/prim-mpi-open-mp-2 100

test-mpi-2-p4-t16: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_2)/prim-mpi-open-mp-2 100

#############
#test 8 processes
test-mpi-2-p8: test-mpi-2-p8-t0 test-mpi-2-p8-t2 test-mpi-2-p8-t4 test-mpi-2-p8-t8 test-mpi-2-p8-t16

#test n threads
test-mpi-2-p8-t0: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	mpirun -np 8 ./$(BUILD_DIR_2)/prim-mpi-2 100

test-mpi-2-p8-t2: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_2)/prim-mpi-open-mp-2 100

test-mpi-2-p8-t4: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_2)/prim-mpi-open-mp-2 100

test-mpi-2-p8-t8: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_2)/prim-mpi-open-mp-2 100

test-mpi-2-p8-t16: $(BUILD_DIR_2)/prim-mpi-open-mp-2
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_2)/prim-mpi-open-mp-2 100
	
#############

# Test cuda

#############
test-cuda-O2: test-cuda-2-t0 test-cuda-2-t2 test-cuda-2-t4 test-cuda-2-t8 test-cuda-2-t16
#test n threads
test-cuda-2-t0: $(BUILD_DIR_2)/prim-cuda-2
	./$(BUILD_DIR_2)/prim-cuda-2 100

test-cuda-2-t2: $(BUILD_DIR_2)/prim-cuda-open-mp-2
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	./$(BUILD_DIR_2)/prim-cuda-open-mp-2 100

test-cuda-2-t4: $(BUILD_DIR_2)/prim-cuda-open-mp-2
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	./$(BUILD_DIR_2)/prim-cuda-open-mp-2 100

test-cuda-2-t8: $(BUILD_DIR_2)/prim-cuda-open-mp-2
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	./$(BUILD_DIR_2)/prim-cuda-open-mp-2 100

test-cuda-2-t16: $(BUILD_DIR_2)/prim-cuda-open-mp-2
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	./$(BUILD_DIR_2)/prim-cuda-open-mp-2 100

#############

# Test prim

#############

#####################################################################################
# Test -O3
test-O3: test-mpi-O3 test-cuda-O3 

#############

# Test mpi

#############
test-mpi-O3: test-mpi-3-p2 test-mpi-3-p4 test-mpi-3-p8
# Test 2 processes
test-mpi-3-p2: test-mpi-3-p2-t0 test-mpi-3-p2-t2 test-mpi-3-p2-t4 test-mpi-3-p2-t8 test-mpi-3-p2-t16
# test n threads
test-mpi-3-p2-t0: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	mpirun -np 2 ./$(BUILD_DIR_3)/prim-mpi-3 100

test-mpi-3-p2-t2: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_3)/prim-mpi-open-mp-3 100

test-mpi-3-p2-t4: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_3)/prim-mpi-open-mp-3 100

test-mpi-3-p2-t8: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_3)/prim-mpi-open-mp-3 100

test-mpi-3-p2-t16: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 2 ./$(BUILD_DIR_3)/prim-mpi-open-mp-3 100

#############
# test 4 processes
test-mpi-3-p4: test-mpi-3-p4-t0 test-mpi-3-p4-t2 test-mpi-3-p4-t4 test-mpi-3-p4-t8 test-mpi-3-p4-t16
#test n threads
test-mpi-3-p4-t0: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	mpirun -np 4 ./$(BUILD_DIR_3)/prim-mpi-3 100

test-mpi-3-p4-t2: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_3)/prim-mpi-open-mp-3 100

test-mpi-3-p4-t4: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_3)/prim-mpi-open-mp-3 100

test-mpi-3-p4-t8: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_3)/prim-mpi-open-mp-3 100

test-mpi-3-p4-t16: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 4 ./$(BUILD_DIR_3)/prim-mpi-open-mp-3 100

#############
#test 8 processes
test-mpi-3-p8: test-mpi-3-p8-t0 test-mpi-3-p8-t2 test-mpi-3-p8-t4 test-mpi-3-p8-t8 test-mpi-3-p8-t16
#test n threads
test-mpi-3-p8-t0: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	mpirun -np 8 ./$(BUILD_DIR_3)/prim-mpi-3 100

test-mpi-3-p8-t2: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_3)/prim-mpi-open-mp-3 100

test-mpi-3-p8-t4: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_3)/prim-mpi-open-mp-3 100

test-mpi-3-p8-t8: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_3)/prim-mpi-open-mp-3 100

test-mpi-3-p8-t16: $(BUILD_DIR_3)/prim-mpi-open-mp-3
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	mpirun -np 8 ./$(BUILD_DIR_3)/prim-mpi-open-mp-3 100

#############

# Test cuda

#############

test-cuda-O3: test-cuda-3-t0 test-cuda-3-t2 test-cuda-3-t4 test-cuda-3-t8 test-cuda-3-t16
#test n threads
test-cuda-3-t0: $(BUILD_DIR_3)/prim-cuda-3
	./$(BUILD_DIR_3)/prim-cuda-3 100

test-cuda-3-t2: $(BUILD_DIR_3)/prim-cuda-open-mp-3
	@export OMP_NUM_THREADS=2 USE_OPENMP=1;
	./$(BUILD_DIR_3)/prim-cuda-open-mp-3 100

test-cuda-3-t4: $(BUILD_DIR_3)/prim-cuda-open-mp-3
	@export OMP_NUM_THREADS=4 USE_OPENMP=1;
	./$(BUILD_DIR_3)/prim-cuda-open-mp-3 100

test-cuda-3-t8: $(BUILD_DIR_3)/prim-cuda-open-mp-3
	@export OMP_NUM_THREADS=8 USE_OPENMP=1;
	./$(BUILD_DIR_3)/prim-cuda-open-mp-3 100

test-cuda-3-t16: $(BUILD_DIR_3)/prim-cuda-open-mp-3
	@export OMP_NUM_THREADS=16 USE_OPENMP=1;
	./$(BUILD_DIR_3)/prim-cuda-open-mp-3 100

#############

# Test prim

#############

test-prim: $(BUILD_DIR_PRIM)/prim
	./$(BUILD_DIR_PRIM)/prim 100

#####################################################################################





	

