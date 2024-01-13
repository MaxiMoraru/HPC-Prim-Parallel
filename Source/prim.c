
/*
#-Moraru, Maximilian Marius, MAT. 0622702167, m.moraru@studenti.unisa.it
#-Algoritmo “Prim”
#-Moscato Francesco, fmoscato@unisa.it
#-Final Projects are assigned to each student. Student shall provide a parallel version of an algorithm with both "OpenMP + MPI" and "OpenMP + Cuda" approaches, comparing results with a known solution on single-processing node. Results and differences shall be discussed for different inputs (type and size). The parallel algorithm used in "OpenMP + MPI" solution could not be the same of the "OpenMP + CUDA" approach.



    prim.c
    Prim's algorithm for finding the minimum spanning tree of a graph

    
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
#include <math.h>
#include <limits.h>
#include <time.h>
#include <stdbool.h>

//#define MATRIXFILE "./Data/matrix-500.txt"

typedef struct Connection {
    int value;
    int v1;
    int v2;
} Connection;


int main(int argc, char *argv[]){

    // Check if the command line argument is provided
    if (argc < 2) {
        printf("Error: No command line argument provided\n");
        return 1;
    }

    //start timer
    clock_t start = clock();
    
    //declare host variables

    FILE *f_matrix;
    FILE *f_time;
    int mSize; // Declare the size of the matrix
    Connection *MST; // Declare MST variable

    // Create a character array to hold the filename
    char matrixFile[50];

    // Format the filename string with the command line argument
    sprintf(matrixFile, "./Data/matrix-%s.txt", argv[1]);

    // Open the file
    f_matrix = fopen(matrixFile, "r");
    if (f_matrix){
        // Read the number of vertices
        fscanf(f_matrix, "%d\n", &mSize);
    }
    else {
        printf("Error: File not found\n");
        return 1;
    }

    // Allocate memory for the matrix
    int *Matrix = (int *)malloc(mSize * mSize * sizeof(int));

    // Read the matrix from the file
    for (int i = 0; i < mSize; i++) {
        for (int j = 0; j < mSize; j++) {
            fscanf(f_matrix, "%d", &Matrix[i * mSize + j]);
        }
    }

    fclose(f_matrix);

    //save the time for load_data
    clock_t load_data_time_stop = clock();


    // Allocate memory for the MST
    MST = (Connection *)malloc(mSize * sizeof(Connection));

    // Initialize the MST
    for (int i = 0; i < mSize; i++) {
        MST[i].value = -1;
    }

    // Set the first vertex as the root
    MST[0].value = 0;
    MST[0].v1 = 0;
    MST[0].v2 = 0;

    // Initialize the minWeight
    int minWeight = 0;

    int  i, j;
    Connection *min;

    min = (Connection *)malloc( mSize * sizeof(Connection));



    for ( int k = 0; k < mSize - 1; ++k){

            for (i = 0; i < mSize; ++i){

                min[i].value = INT_MAX;
                min[i].v1 = -1;
                min[i].v2 = -1;

                if (MST[i].value != -1) {

                    for (j = 0; j < mSize; ++j){

                        if (MST[j].value == -1) {
                           
                            if ( Matrix[mSize*i+j] < min[i].value && Matrix[mSize*i+j] != 0){
                                        
                                min[i].value = Matrix[mSize*i+j];
                                min[i].v1 = i;
                                min[i].v2 = j;
                                        
                            }
                            
                        }
                    }
                }
            }
        

        // find global min
        Connection globalMin;
        globalMin.value = INT_MAX;
        
        for (int i = 0; i < mSize; ++i) {
            if(min[i].value != INT_MAX){
                if (min[i].value < globalMin.value) {
                    globalMin = min[i];
                }
            }
        }


        // update MST
        MST[globalMin.v2] = globalMin;
        minWeight += globalMin.value;
        

        
    }

    //stop timer
    clock_t stop = clock();

    //calculate the time for load_data, the time for the algorithm and the total time
    double load_data_time = (double)(load_data_time_stop - start) / CLOCKS_PER_SEC;
    double calc_time = (double)(stop - load_data_time_stop) / CLOCKS_PER_SEC;
    double total_time = (double)(stop - start) / CLOCKS_PER_SEC;

    FILE *f_result;

    // Open the result file and write the results
    f_result = fopen("./Data/ResultPrim.txt", "w");
    fprintf(f_result,"The min minWeight is %d\n", minWeight);
    for (int i = 0; i < mSize; ++i){
        fprintf(f_result,"V%d connects: ", i);
        bool isConnected = false;
        for (int j = 0; j < mSize; ++j) {
            if ((MST[j].v1 == i || MST[j].v2 == i) && j != i) {
                fprintf(f_result, "%d ", j);
                isConnected = true;
            }
        }
        if (!isConnected) {
            fprintf(f_result, "none");
        }
        fprintf(f_result, "\n");
    }
    fclose(f_result);

    //print the time for load_data, the time for the algorithm and the total time in cvs file
    f_time = fopen("./Data/TimePrim.txt", "a+");
    //if the file is empty write the columns names as cvs format : Threads, Processors, Vertices, Load data time, Calculation time, Total time\n"
    if(ftell(f_time) == 0){
       // fprintf(f_time, "Vertices, Load data time, Calculation time, Total time\n");
    }
    //now print the values
    fprintf(f_time, "%d, %f, %f, %f\n", mSize, load_data_time, calc_time, total_time);
    fclose(f_time);



    // Free the memory

    free(Matrix);
    free(MST);
    free(min);



    return 0;
}

