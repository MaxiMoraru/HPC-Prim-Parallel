/*
#-Moraru, Maximilian Marius, MAT. 0622702167, m.moraru@studenti.unisa.it
#-Algoritmo “Prim”
#-Moscato Francesco, fmoscato@unisa.it
#-Final Projects are assigned to each student. Student shall provide a parallel version of an algorithm with both "OpenMP + MPI" and "OpenMP + Cuda" approaches, comparing results with a known solution on single-processing node. Results and differences shall be discussed for different inputs (type and size). The parallel algorithm used in "OpenMP + MPI" solution could not be the same of the "OpenMP + CUDA" approach.



   
    
#Copyright (C) <2024>  <Maximilian Marius Moraru>

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


CC = gcc
CFLAGS = -Wall -Wextra -O2

all: generate-matrix-c

generate-matrix-c: generate-matrix.c
	$(CC) $(CFLAGS) -o $@ $<

test: generate-matrix
	./generate-matrix

clean:
	rm -f generate-matrix
