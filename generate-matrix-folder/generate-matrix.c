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
    generateMatrix(1000);

    return 0;
}
