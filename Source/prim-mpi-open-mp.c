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



#define MATRIXFILE "./Data/matrix-1000.txt"

typedef struct Connection {
    int value;
    int v1;
    int v2;
} Connection;


#pragma omp declare reduction(minimum : Connection : omp_out = omp_in.value < omp_out.value ? omp_in : omp_out) initializer (omp_priv=omp_orig)

int size; // number of processors
int rank; // rank of each processor
int* MatrixChunk; // chunk of matrix for each processor
int mSize; // number of vertices
int* displs; // displacements
int* sendcounts; // sendcounts
//typedef struct { int v1; int v2; } Edge; // edge
Connection* MST; // minimum spanning tree
int minWeight; // minimum weight
FILE *f_matrix; // file of matrix
FILE *f_time; // file of time 
FILE *f_result; // file of result

typedef struct ConnRank {
    Connection conn;
    int rank;
} ConnRank;

//custom mpi reduce function
void minConnRank(void *in, void *inout, int *len, MPI_Datatype *datatype){
    ConnRank *inConnRank = (ConnRank *)in;
    ConnRank *inoutConnRank = (ConnRank *)inout;

    if (inConnRank->conn.value < inoutConnRank->conn.value) {
        *inoutConnRank = *inConnRank;
    }
}


int main(int argc,char *argv[]){

    MPI_Init ( &argc, &argv );
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    MPI_Comm_size ( MPI_COMM_WORLD, &size );

    // Define the MPI datatype for the ConnRank struct
    MPI_Datatype connRankType;
    MPI_Type_contiguous(4, MPI_INT, &connRankType);
    MPI_Type_commit(&connRankType);

    //create custom mpi reduce function
    MPI_Op minConnRankOp;
    MPI_Op_create(minConnRank, 1, &minConnRankOp);



    /************************************************/
    // read the number of vertices from file
    /************************************************/
    if (rank==0){
        f_matrix = fopen(MATRIXFILE, "r");
        if (f_matrix){
            fscanf(f_matrix, "%d\n", &mSize);
        }
        else {
            printf("File not found.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fclose(f_matrix);
    }

    MPI_Bcast(&mSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /************************************************/
    // calculate the displacements and sendcounts
    /************************************************/
    
    int i,j,k;       

    displs = (int*)malloc( size * sizeof(int));   
    sendcounts = (int*)malloc(size * sizeof(int));

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
        f_matrix = fopen(MATRIXFILE, "r");
        if (!f_matrix) {
            fprintf(stderr, "Error opening file.\n");
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


    MST = (Connection*)malloc(mSize * sizeof(Connection)); // max size is number of vertices

    for ( i = 0; i < mSize; ++i){
        MST[i].value = -1;
    }
    
    double start; 

    start = MPI_Wtime(); // start to calculate running time
        
    MST[0].value = 0;
    MST[0].v1 = 0;
    MST[0].v2 = 0;
    
    minWeight = 0;

    Connection *min;
    min = (Connection *)malloc( mSize * sizeof(Connection));


    //cannot use parallel for here because prim is a sequential algorithm
    for ( k = 0; k < mSize - 1; ++k){

        Connection local;
        local.value = INT_MAX;
        local.v1 = -1;
        local.v2 = -1;


        #pragma omp parallel for shared(min, MST, MatrixChunk) private(i, j)
        for ( i = 0; i < sendcounts[rank]; ++i){

            min[i].value = INT_MAX;
            min[i].v1 = -1;
            min[i].v2 = -1;

            if (MST[i + displs[rank]].value != -1) {

                for ( j = 0; j < mSize; ++j){

                    if (MST[j].value == -1) {

                        if ( MatrixChunk[mSize*i+j] < min[i].value && MatrixChunk[mSize*i+j] != 0){

                            min[i].value = MatrixChunk[mSize*i+j];
                            min[i].v1 = i+displs[rank];
                            min[i].v2 = j;
                            
                        }
                            
                    }
                }
            }
        }

        //find local min

        #pragma omp parallel for reduction(minimum: local)
        for(int i = 0; i < sendcounts[rank]; i++){
            if(min[i].value != INT_MAX){
                if (min[i].value < local.value) {
                    local = min[i];
                }
            }
        }


        ConnRank localMin;
        localMin.rank = rank;
        localMin.conn = local;
        
        ConnRank globalMin;
        //find global min
        
        MPI_Allreduce(&localMin, &globalMin, 1, connRankType, minConnRankOp, MPI_COMM_WORLD);

        MST[globalMin.conn.v2] = globalMin.conn;
        minWeight += globalMin.conn.value;
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
                printf("MST[%d].v1 = %d\n", j, MST[j].v1);
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