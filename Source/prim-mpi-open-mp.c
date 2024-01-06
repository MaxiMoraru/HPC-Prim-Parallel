#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <stdbool.h>

int size; // number of processors
int rank; // rank of each processor
int* MatrixChunk; // chunk of matrix for each processor
int mSize; // number of vertices
int* displs; // displacements
int* sendcounts; // sendcounts
typedef struct { int v1; int v2; } Edge; // edge
int* MST; // minimum spanning tree
int minWeight; // minimum weight
FILE *f_matrix; // file of matrix
FILE *f_time; // file of time 
FILE *f_result; // file of result

int main(int argc,char *argv[]){

    MPI_Init ( &argc, &argv );
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    MPI_Comm_size ( MPI_COMM_WORLD, &size );


    /************************************************/
    // read the number of vertices from file
    /************************************************/
    if (rank==0){
        f_matrix = fopen("./Data/matrix-100.txt", "r");
        if (f_matrix){
            fscanf(f_matrix, "%d\n", &mSize);
        }
        else {
            printf("File matrix-100.txt not found.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(f_matrix);
    }


    MPI_Bcast(&mSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /************************************************/
    // calculate the displacements and sendcounts
    /************************************************/
    
    int i,j,k;       

    displs = (int*)malloc(sizeof(int) * size);   
    sendcounts = (int*)malloc(sizeof(int) * size);

    displs[0] = 0;
    sendcounts[0] = mSize / size;

    int remains = size - (mSize % size); // if number of vertices is not a multiply of number of processors

    for (i = 1; i < (size - remains); ++i) {
        sendcounts[i] = sendcounts[0];
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
    for (i = (size - remains); i < size; ++i){
        sendcounts[i] = sendcounts[0] + 1;
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    /************************************************/
    // allocate the matrix chunk for each processor
    // read the matrix from file
    // send the matrix chunk to each processor
    /************************************************/


    MatrixChunk = (int *)malloc(sendcounts[rank] * mSize * sizeof(int));

    if (rank == 0) {
        //allocate the matrix chunk to read the matrix from the file
        int *MatrixChunkRead = (int *)malloc(sendcounts[size-1] * mSize * sizeof(int));

        // Read the matrix from the file on the root process
        f_matrix = fopen("./Data/matrix-100.txt", "r");
        if (!f_matrix) {
            fprintf(stderr, "Error opening file: %s\n", "../Data/matrix-100.txt");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        // Skip the first value (already read the number of rows)
        fscanf(f_matrix, "%*d");

        // Read and send each chunk of the matrix to its respective process
        for (int i = 0; i < size; i++) {
            // Read the matrix chunk from the file
            for (int j = 0; j < sendcounts[i]; j++) {
                // Read each row of the chunk
                for (int k = 0; k < mSize; k++) {
                    if(i == 0){
                        fscanf(f_matrix, "%d", &MatrixChunk[j * mSize + k]);
                    }else{
                        fscanf(f_matrix, "%d", &MatrixChunkRead[j * mSize + k]);
                    }
                }
            }
            if(i != 0){
                MPI_Send(MatrixChunkRead, sendcounts[i] * mSize, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }

        free(MatrixChunkRead);

        fclose(f_matrix);

    } else {
        // Receive the local matrix on other processes
        MPI_Recv(MatrixChunk, sendcounts[rank] * mSize, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    /************************************************/
    // calculate the minimum spanning tree
    /************************************************/


    MST = (int*)malloc(sizeof(int)*mSize); // max size is number of vertices

    for ( i = 0; i < mSize; ++i){
        MST[i] = -1;
    }
    
    double start; 

    start = MPI_Wtime(); // start to calculate running time
        
    MST[0] = 0;
    minWeight = 0;

    int min;
    int v1 = 0;
    int v2 = 0;

    struct { int min; int rank; } minRow, row;
    Edge edge;


    //cannot use parallel for here because prim is a sequential algorithm
    for ( k = 0; k < mSize - 1; ++k){
        min = INT_MAX;

        #pragma omp parallel for 
        for ( i = 0; i < sendcounts[rank]; ++i){

            if (MST[i + displs[rank]] != -1) {

                for ( j = 0; j < mSize; ++j){

                    if (MST[j] == -1) {

                        #pragma omp critical
                        {
                            //if the MatrixChunk[mSize*i+j] is less than min value
                            if ( MatrixChunk[mSize*i+j] < min && MatrixChunk[mSize*i+j] != 0){
                                    
                                min = MatrixChunk[mSize*i+j];
                                v2 = j; // change the current edge
                                v1 = i;
                                    
                            }
                        }
                    }
                }
            }
        }
        row.min = min;
        row.rank = rank; // the rank of min in row
        // each proc have to send the min row to others 
        MPI_Allreduce(&row, &minRow, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD); 
        edge.v1 = v1 + displs[rank];
        edge.v2 = v2;
        MPI_Bcast(&edge, 1, MPI_2INT, minRow.rank, MPI_COMM_WORLD);

        MST[edge.v2] = edge.v1;
        minWeight += minRow.min;
    }
    
    /************************************************/
    // print the result
    /************************************************/

    double finish, calc_time; 
    finish = MPI_Wtime();
    calc_time = finish-start;

    if (rank == 0){
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

        f_time = fopen("./Data/Time.txt", "a+");
        fprintf(f_time, "\n Number of processors: %d\n Number of vertices: %d\n Time of execution: %f\n Total Weight: %d\n\n", size, mSize ,calc_time, minWeight);
        fclose(f_time);
        
    }

    free(MatrixChunk);
    free(MST);
    free(displs);
    free(sendcounts);


    MPI_Finalize();
    return 0;

    }