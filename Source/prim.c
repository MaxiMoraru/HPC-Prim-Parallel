#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <stdbool.h>


int main(int argc,char *argv[]){
    
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

    int v1, v2, min;

    for ( int k = 0; k < mSize - 1; ++k){
        min = INT_MAX;

        for (int i = 0; i < mSize; ++i){

            if (MST[i] != -1) {

                for ( int j = 0; j < mSize; ++j){

                    if (MST[j] == -1) {

                        //if the MatrixChunk[mSize*i+j] is less than min value
                        if ( Matrix[mSize*i+j] < min && Matrix[mSize*i+j] != 0){
                                    
                            min = Matrix[mSize*i+j];
                            v2 = j; // change the current edge
                            v1 = i;
                            
                        }
                    }
                }
            }

        }

        // Add the new vertex to the MST
        MST[v2] = v1;
        
        // Add the weight of the edge to the minWeight
        minWeight += min;
        

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