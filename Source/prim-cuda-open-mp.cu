#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 1024

__global__ void findMin(int *Matrix, int *MST, int *min, int *v1, int *v2, int mSize) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < mSize; i += stride) {
        if (MST[i] != -1) {
            for (int j = 0; j < mSize; ++j) {
                if (MST[j] == -1 && Matrix[mSize*i+j] < min[index] && Matrix[mSize*i+j] != 0) {
                    min[index] = Matrix[mSize*i+j];
                    v1[index] = i;
                    v2[index] = j;
                }
            }
        }
    }
}

int main(int argc,char *argv[]){
    // ... omitted code for reading the matrix and initializing variables ...

    FILE *f_matrix;
    int mSize, mSize;
    int *MST; // Declare MST variable

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

    int *dev_Matrix, *dev_MST, *dev_min, *dev_v1, *dev_v2;
    MST = (int *)malloc(mSize * sizeof(int)); // Allocate memory for MST

    cudaMalloc((void**)&dev_Matrix, mSize * mSize * sizeof(int));
    cudaMalloc((void**)&dev_MST, mSize * sizeof(int));
    cudaMalloc((void**)&dev_min, mSize * sizeof(int));
    cudaMalloc((void**)&dev_v1, mSize * sizeof(int));
    cudaMalloc((void**)&dev_v2, mSize * sizeof(int));

    cudaMemcpy(dev_Matrix, Matrix, mSize * mSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_MST, MST, mSize * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (mSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int *min = (int *)malloc(mSize * sizeof(int));
    int *v1 = (int *)malloc(mSize * sizeof(int));
    int *v2 = (int *)malloc(mSize * sizeof(int));
    int minWeight = 0;

    for (int k = 0; k < mSize - 1; ++k) {
        findMin<<<blocks, THREADS_PER_BLOCK>>>(dev_Matrix, dev_MST, dev_min, dev_v1, dev_v2, mSize);

        cudaMemcpy(min, dev_min, mSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(v1, dev_v1, mSize * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(v2, dev_v2, mSize * sizeof(int), cudaMemcpyDeviceToHost);

        int globalMin = INT_MAX;
        int globalV1, globalV2;

        #pragma omp parallel for reduction(min:globalMin)
        for (int i = 0; i < mSize; ++i) {
            if (min[i] < globalMin) {
                globalMin = min[i];
                globalV1 = v1[i];
                globalV2 = v2[i];
            }
        }

        MST[globalV2] = globalV1;
        minWeight += globalMin;

        cudaMemcpy(dev_MST, MST, mSize * sizeof(int), cudaMemcpyHostToDevice);
    }

    // ... omitted code for writing the result and cleaning up ...

     FILE *f_result, *f_time;
    int minWeight = 0; // You need to calculate this

    // Open the result file and write the results
    f_result = fopen("./Data/Result.txt", "w");
    fprintf(f_result,"The min minWeight is %d\n ", minWeight);
    for (int i = 0; i < mSize; ++i){
        fprintf(f_result,"Edge %d %d\n",i, MST[i]);
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
    cudaFree(dev_v1);
    cudaFree(dev_v2);

    return 0;
}