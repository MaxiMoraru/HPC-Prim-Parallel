#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <stdbool.h>

#define MATRIXFILE "./Data/matrix-100.txt"

typedef struct Connection {
    int value;
    int v1;
    int v2;
} Connection;

#pragma omp declare reduction(minimum : Connection : omp_out = omp_in.value < omp_out.value ? omp_in : omp_out) initializer (omp_priv=omp_orig)


int main(){
    
    //declare host variables

    FILE *f_matrix;
    int mSize; // Declare the size of the matrix
    Connection *MST; // Declare MST variable

    // Open the file
    f_matrix = fopen(MATRIXFILE, "r");
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


            #pragma omp parallel for shared(min, MST, Matrix) private(i, j)
        
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
        
        
        #pragma omp parallel for reduction(minimum: globalMin)
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


    FILE *f_result;

    // Open the result file and write the results
    f_result = fopen("./Data/Result.txt", "w");
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

    // Free the memory

    free(Matrix);
    free(MST);
    free(min);



    return 0;
}

