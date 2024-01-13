
/*

#-Moraru, Maximilian Marius, MAT. 0622702167, m.moraru@studenti.unisa.it
#-Algoritmo “Prim”
#-Moscato Francesco, fmoscato@unisa.it
#-Final Projects are assigned to each student. Student shall provide a parallel version of an algorithm with both "OpenMP + MPI" and "OpenMP + Cuda" approaches, comparing results with a known solution on single-processing node. Results and differences shall be discussed for different inputs (type and size). The parallel algorithm used in "OpenMP + MPI" solution could not be the same of the "OpenMP + CUDA" approach.


    This program finds the minimum spanning tree of a graph using Prim's algorithm.
    The program uses CUDA to parallelize the algorithm.
    The program takes as input the number of vertices and the name of the file containing the adjacency matrix.
    The program outputs the minimum spanning tree and the minimum weight.
    The program also outputs the execution time in a file named TimeCuda.txt.
    The program also outputs the minimum spanning tree in a file named ResultCuda.txt.
    The program also outputs the minimum weight in a file named ResultCuda.txt.
    The program also outputs the execution time in a file named TimeCuda.txt.
    The program also outputs the number of threads, the number of processors and the number of vertices in a file named TimeCuda.txt.
    The program also outputs the load data time, the calculation time and the total time in a file named TimeCuda.txt.
   
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
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <cuda.h>

//#define MATRIXFILE "./Data/matrix-500.txt"

#define THREADS_PER_BLOCK 1024



typedef struct Connection {
    int value;
    int v1;
    int v2;
} Connection;

#pragma omp declare reduction(minimum : Connection : omp_out = omp_in.value < omp_out.value ? omp_in : omp_out) initializer (omp_priv=omp_orig)

__global__ void findMin(int *Matrix, Connection *MST, Connection *min, int mSize) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < mSize; i += stride) {
        min[index].value = INT_MAX;
        min[index].v1 = -1;
        min[index].v2 = -1;
        if (MST[i].value != -1) {
            for (int j = 0; j < mSize; ++j) {
                if (MST[j].value == -1 && Matrix[mSize*i+j] < min[index].value && Matrix[mSize*i+j] != 0) {
                    min[index].value = Matrix[mSize*i+j];
                    min[index].v1 = i;
                    min[index].v2 = j;
                }
            }
        }
    }
}



int main(int argc,char *argv[]){

    // Check if the command line argument is provided
    if (argc < 2) {
        printf("Error: No command line argument provided\n");
        return 1;
    }

    //start timer
    clock_t start = clock();

    /*****************************************************/
    // get the matrix from the file
    /*****************************************************/



    FILE *f_matrix;
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
        printf("File not found.\n");
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

    /*****************************************************/
    // find the MST
    /*****************************************************/


    //declare device variables
    int *dev_Matrix;
    Connection *dev_MST, *dev_min;

    MST = (Connection *)malloc(mSize * sizeof(Connection)); // Allocate memory for MST

    // Initialize MST
    for (int i = 0; i < mSize; ++i) {
        MST[i].value = -1;
    }

    // Set the first vertex as the root
    MST[0].value = 0;
    MST[0].v1 = 0;
    MST[0].v2 = 0;

    // Initialize the minWeight
    int minWeight = 0;

    // allocate memory on device
    cudaMalloc((void**)&dev_Matrix, mSize * mSize * sizeof(int));
    cudaMalloc((void**)&dev_MST, mSize * sizeof(Connection));
    cudaMalloc((void**)&dev_min, mSize * sizeof(Connection));

    // copy data from host to device
    cudaMemcpy(dev_Matrix, Matrix, mSize * mSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_MST, MST, mSize * sizeof(Connection), cudaMemcpyHostToDevice);

    //declare blocks
    int blocks = (mSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    //declare host variables
    Connection *min = (Connection *)malloc(mSize * sizeof(Connection));
    

    for (int k = 0; k < mSize - 1; ++k) {

        //call kernel
        findMin<<<blocks, THREADS_PER_BLOCK>>>(dev_Matrix, dev_MST, dev_min, mSize);

        //check for errors
        cudaError_t errSync  = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) 
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        if (errAsync != cudaSuccess)
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));


        //copy data from device to host
        cudaMemcpy(min, dev_min, mSize * sizeof(Connection), cudaMemcpyDeviceToHost);
        
        //print the min array indicating the min value it contains, the vertices its connected to and the vertices its connected from and the iteration number k it is in while also ignoring the INT_MAX values
        
    
        

        //find global min
        Connection globalMin;

        globalMin.value = INT_MAX;



        //parallelize the for using openmp

        #pragma omp parallel for reduction(minimum: globalMin)

        for (int i = 0; i < k+1; ++i) {
            if (min[i].value < globalMin.value) {
                globalMin = min[i];
            }
        }
        MST[globalMin.v2] = globalMin;
        minWeight += globalMin.value;

        //update MST on device
        cudaMemcpy(dev_MST, MST, mSize * sizeof(Connection), cudaMemcpyHostToDevice);
    }





    /*****************************************************/
    // write the result to the file
    /*****************************************************/

    //stop timer
    clock_t stop = clock();

    //calculate the time for load_data, the time for the algorithm and the total time
    double load_data_time = (double)(load_data_time_stop - start) / CLOCKS_PER_SEC;
    double calc_time = (double)(stop - load_data_time_stop) / CLOCKS_PER_SEC;
    double total_time = (double)(stop - start) / CLOCKS_PER_SEC;
    


    FILE *f_result, *f_time;

    // Open the result file and write the results
    f_result = fopen("./Data/ResultCuda.txt", "w");
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

    // Open the time file and write the execution time
    f_time = fopen("./Data/TimeCuda.txt", "a+");
    //if the file is empty write the columns names as cvs format : Threads, Processors, Vertices, Load data time, Calculation time, Total time\n"
    if(ftell(f_time) == 0){
       // fprintf(f_time, "Vertices, Threads, Load data time, Calculation time, Total time\n");
    }
    //now print the values
    
    #ifdef USE_OPENMP
        fprintf(f_time, "%d, %d, %f, %f, %f\n", mSize, omp_get_max_threads(), load_data_time, calc_time , total_time);
    #else
        fprintf(f_time, "%d, %d, %f, %f, %f\n", mSize, 0, load_data_time, calc_time , total_time);    
    #endif
    
    fclose(f_time);

    //free memory
    cudaFree(dev_Matrix);
    cudaFree(dev_MST);
    cudaFree(dev_min);
    free(Matrix);
    free(MST);
    free(min);


    return 0;
}