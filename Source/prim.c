#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <stdbool.h>

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

                if (MST[i].value != -1) {

                    for (j = 0; j < mSize; ++j){

                        if (MST[j].value == -1) {

                            //if the MatrixChunk[mSize*i+j] is less than min value
                            
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
            if (min[i].value < globalMin.value) {
                globalMin = min[i];
            }
        }

        // update MST
        MST[globalMin.v2] = globalMin;
        minWeight += globalMin.value;
        /*
            //print the min array indicating the min value it contains, the vertices its connected to and the vertices its connected from and the iteration number k it is in while also ignoring the INT_MAX values
                if(k < 10 ){
                    printf("Iteration %d\n", k);
                    //print the mst array
                    for (int i = 0; i < mSize; ++i) {
                        if(MST[i] != -1){
                            printf("%d - ", i);
                        }
                    }
                    printf("\n");
                    for (int i = 0; i < mSize; ++i) {
                        if (min[i] != INT_MAX) {
                            printf("Min: %d, v1: %d, v2: %d\n", min[i], v1[i], v2[i]);
                        }
                    }
                }
        */

    }


    
    // ... omitted code for writing the result and cleaning up ...

     FILE *f_result;

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

    // Free the memory

    free(Matrix);
    free(MST);
    free(min);



    return 0;
}




/*

    // find global min
        int globalMin = INT_MAX;
        int globalV1, globalV2;

        struct MinIndex {
            int value;
            int v1;
            int v2;
        };

        struct MinIndex minIndex;
        minIndex.value = INT_MAX;


        #pragma omp declare reduction(minimum : struct MinIndex : omp_out = omp_in.value < omp_out.value ? omp_in : omp_out)

        #pragma omp parallel for reduction(minimum: minIndex)
        for (int i = 0; i < mSize; ++i) {
            if (min[i] < minIndex.value) {
                minIndex.value = min[i];
                minIndex.v1 = v1[i];
                minIndex.v2 = v2[i];
            }
        }

        globalMin = minIndex.value;
        globalV1 = minIndex.v1;
        globalV2 = minIndex.v2;

        //update MST
        MST[globalV2] = globalV1;
        minWeight += globalMin;
        

*/