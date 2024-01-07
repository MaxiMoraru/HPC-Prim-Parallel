#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>

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
    
    /*****************************************************/
    // get the matrix from the file
    /*****************************************************/

    FILE *f_matrix;
    int mSize; // Declare the size of the matrix
    Connection *MST; // Declare MST variable

    // Open the file
    f_matrix = fopen("./Data/matrix-100.txt", "r");
    if (f_matrix){
        // Read the number of vertices
        fscanf(f_matrix, "%d\n", &mSize);
    }
    else {
        printf("File matrix-100.txt not found.\n");
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

        //wait for kernel to finish
        cudaDeviceSynchronize();

        //copy data from device to host
        cudaMemcpy(min, dev_min, mSize * sizeof(Connection), cudaMemcpyDeviceToHost);
        
        //print the min array indicating the min value it contains, the vertices its connected to and the vertices its connected from and the iteration number k it is in while also ignoring the INT_MAX values

        //find global min
        Connection globalMin;

        globalMin.value = INT_MAX;



        //parallelize the for using openmp

        #pragma omp parallel for reduction(minimum: globalMin)

        for (int i = 0; i < mSize; ++i) {
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

     FILE *f_result, *f_time;

    // Open the result file and write the results
    f_result = fopen("./Data/Result.txt", "w");
    fprintf(f_result,"The min minWeight is %d\n", minWeight);
    for (int i = 0; i < mSize; ++i){
        fprintf(f_result,"V%d connects: ", i);
        bool isConnected = false;
        for (int j = 0; j < mSize; ++j) {
            if (MST[j].v1 == i || MST[j].v2 == i) {
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
    f_time = fopen("./Data/Time.txt", "a+");
    double calc_time = 0; // You need to calculate this
    fprintf(f_time, "\n Number of vertices: %d\n Time of execution: %f\n Total Weight: %d\n\n", mSize ,calc_time, minWeight);
    fclose(f_time);


    cudaFree(dev_Matrix);
    cudaFree(dev_MST);
    cudaFree(dev_min);
    free(Matrix);
    free(MST);
    free(min);


    return 0;
}