#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <stdbool.h>


int main(){
    
    //declare host variables

    FILE *f_matrix;
    int mSize; // Declare the size of the matrix
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

    // Allocate memory for the MST
    MST = (int *)malloc(mSize * sizeof(int));

    // Initialize the MST
    for (int i = 0; i < mSize; i++) {
        MST[i] = -1;
    }

    // Set the first vertex as the root
    MST[0] = 0;

    // Initialize the minWeight
    int minWeight = 0;

    int *v1, *v2, *min, i, j;

    min = (int *)malloc( mSize * sizeof(int));
    v1 = (int *)malloc( mSize * sizeof(int));
    v2 = (int *)malloc( mSize * sizeof(int));


    for ( int k = 0; k < mSize - 1; ++k){


        #pragma omp parallel for shared(min, v1, v2, MST, Matrix) private(i, j)
        
            for (i = 0; i < mSize; ++i){

                min[i] = INT_MAX;

                if (MST[i] != -1) {

                    for (j = 0; j < mSize; ++j){

                        if (MST[j] == -1) {

                            //if the MatrixChunk[mSize*i+j] is less than min value
                            
                            if ( Matrix[mSize*i+j] < min[i] && Matrix[mSize*i+j] != 0){
                                    
                                min[i] = Matrix[mSize*i+j];
                                v1[i] = i; // change the current edge
                                v2[i] = j;
                                    
                            }
                            
                        }
                    }
                }
            }
        

        // find global min
        int globalMin = INT_MAX;
        int globalV1, globalV2;

        for (int i = 0; i < mSize; ++i) {
            if (min[i] < globalMin) {
                globalMin = min[i];
                globalV1 = v1[i];
                globalV2 = v2[i];
            }
        }

        // update MST
        MST[globalV2] = globalV1;
        minWeight += globalMin;
        
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
            if (MST[j] == i) {
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




    return 0;
}