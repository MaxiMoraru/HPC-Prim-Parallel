
/*

#-Moraru, Maximilian Marius, MAT. 0622702167, m.moraru@studenti.unisa.it
#-Algoritmo “Prim”
#-Moscato Francesco, fmoscato@unisa.it
#-Final Projects are assigned to each student. Student shall provide a parallel version of an algorithm with both "OpenMP + MPI" and "OpenMP + Cuda" approaches, comparing results with a known solution on single-processing node. Results and differences shall be discussed for different inputs (type and size). The parallel algorithm used in "OpenMP + MPI" solution could not be the same of the "OpenMP + CUDA" approach.


#this program generates a matrix of size n*n and writes it to a file. the matrix is symmetric and the diagonal from top right to bottom left is 0. the matrix is generated entirery making sure that the (i,j) element is the same as the (j,i) element


   
Copyright (C) <2024>  <Maximilian Marius Moraru>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void generateMatrix(int n) {
    // Create the filename
    char filename[100];
    sprintf(filename, "../Data/matrix-%d.txt", n);

    // Open the file for writing
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file for writing.\n");
        return;
    }

    // Write the number of vertices in the first row
    fprintf(file, "%d\n", n);

    // Seed the random number generator
    srand(time(NULL));

    //generate the matrix inside a variable and then write it to the file. the matrix should be symmetric and the diagonal from top right to bottom left should be 0 and the matrix should be generate entirery making sure that the (i,j) element is the same as the (j,i) element
    int i, j;

    int matrix[n][n];

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if(j == i) {
                matrix[i][j] = 0;
            } else {
                if(j > i) {
                    matrix[i][j] = rand() % 100;
                } else {
                    matrix[i][j] = matrix[j][i];
                }
            }
            fprintf(file, "%d ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }
    /*
    // Write the matrix to the file
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            fprintf(file, "%d ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }
    */
    // Close the file
    fclose(file);
}

int main() {

    // Generate the matrices
    generateMatrix(500);

    return 0;
}
