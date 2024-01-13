/*

#-Moraru, Maximilian Marius, MAT. 0622702167, m.moraru@studenti.unisa.it
#-Algoritmo “Prim”
#-Moscato Francesco, fmoscato@unisa.it
#-Final Projects are assigned to each student. Student shall provide a parallel version of an algorithm with both "OpenMP + MPI" and "OpenMP + Cuda" approaches, comparing results with a known solution on single-processing node. Results and differences shall be discussed for different inputs (type and size). The parallel algorithm used in "OpenMP + MPI" solution could not be the same of the "OpenMP + CUDA" approach.


this program is the parallel version of prim algorithm using mpi and openmp
the program is run with the command mpirun -np <number of processors> ./prim-mpi-open-mp <number of vertices>
the program will read the matrix from the file Data/matrix-<number of vertices>.txt
the program will write the result in the file Data/ResultMpi.txt
the program will write the time in the file Data/TimeMpi.txt
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


#define MATRIXFILE "./Data/matrix-500.txt"

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
double program_time_start;
double load_data_time_stop;
double program_time_stop;

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

    // Check if the command line argument is provided
    if (argc < 2) {
        printf("Error: No command line argument provided\n");
        return 1;
    }

    // Create a character array to hold the filename
    char matrixFile[50];

    // Format the filename string with the command line argument
    sprintf(matrixFile, "./Data/matrix-%s.txt", argv[1]);

    

    MPI_Init ( &argc, &argv );
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    MPI_Comm_size ( MPI_COMM_WORLD, &size );

    //start the timer
    if(rank == 0){
        program_time_start = MPI_Wtime();
    }

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
        f_matrix = fopen(matrixFile, "r");
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

    //save time after loading data
    if(rank == 0){
        load_data_time_stop = MPI_Wtime();
    }
    
    /************************************************/
    // calculate the minimum spanning tree
    /************************************************/


    MST = (Connection*)malloc(mSize * sizeof(Connection)); // max size is number of vertices

    for ( i = 0; i < mSize; ++i){
        MST[i].value = -1;
    }
    
        
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
        //check if the mst added is the same as the value from the matrix it represents if the rank is the same as the rank of the vertex
        /*
        if(rank == globalMin.rank){
            if(MatrixChunk[mSize*(globalMin.conn.v1-displs[rank]) + globalMin.conn.v2] != globalMin.conn.value){
                printf("Error: MST value is not the same as the value from the matrix it represents\n");
            }else{
                printf("OK\n");
            }
        }
        */
        
    }
    
    /************************************************/
    // print the result
    /************************************************/

    //stop the timer
    if(rank == 0){
        program_time_stop = MPI_Wtime();
    } 
    
    if (rank == 0){

        //calculate the time of execution
        double total_time = program_time_stop - program_time_start;
        double load_data_time = load_data_time_stop - program_time_start;
        double calc_time = program_time_stop - load_data_time_stop;
    

        // Open the result file and write the results
        f_result = fopen("./Data/ResultMpi.txt", "w");
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


        f_time = fopen("./Data/TimeMpi.txt", "a+");
        //if the file is empty write the columns names as cvs format : Threads, Processors, Vertices, Load data time, Calculation time, Total time\n"
        if(ftell(f_time) == 0){
            //fprintf(f_time, "Vertices, Processors, Threads, Load data time, Calculation time, Total time\n");
        }
        //now print the values
        
        
        #ifdef USE_OPENMP
            fprintf(f_time, "%d, %d, %d, %f, %f, %f\n", mSize, size, omp_get_max_threads(), load_data_time, calc_time , total_time);
        #else
            fprintf(f_time, "%d, %d, %d, %f, %f, %f\n", mSize, size, 0, load_data_time, calc_time , total_time);
        #endif
        fclose(f_time);

    }

    free(MatrixChunk);
    free(MST);
    free(displs);
    free(sendcounts);



    MPI_Finalize();
    return 0;

    }


