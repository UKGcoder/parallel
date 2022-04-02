#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "mpi.h"

double t = 0.00001;
double eps = 1e-7;

int main(int argc, char **argv) {
    int N =10000;
    int flag = 1;
    double start_time;
    double end_time;
    int process_count;
    int process_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    if(process_count>N){
        printf("Count of processes is more than N\n");
        exit(0);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    double* x = (double*)malloc(sizeof(double) * N);
    printf("I'm %d from %d processes\n", process_rank, process_count);
    int resN= N % process_count;
    if(resN!=0){
        N+=process_count-resN;
    }else{
        resN=process_count;
    }
    int number_of_elements = N / process_count;

    double* part_A = (double*)malloc(sizeof(double) * N * number_of_elements);
    double* part_b = (double*)malloc(sizeof(double) * number_of_elements);
    double* part_x = (double*)malloc(sizeof(double) * number_of_elements);
    double* part_sum = (double*)malloc(sizeof(double) * number_of_elements);
    start_time = MPI_Wtime();

    for(int i = 0; i< N;i++){
        for(int n = 0;n<number_of_elements;n++){
            part_A[n*N+i]=1;
            if(i==(process_rank+n)){
                part_A[n*N+i]=2;
            }
            part_sum[n]=0;
            part_x[n]=0;
            part_b[n]=N+1;
        }
    }

    while (flag) {
        double norm = 0;
        double norm_b = 0;
        for (int i = 0; i < number_of_elements; ++i) {
            double sum = 0;
            for (int j = 0; j <N; ++j) {
                sum += part_A[i * N + j] * part_x[i];
            }
            part_sum[i]=sum-part_b[i];
            norm += part_sum[i] * part_sum[i];
            norm_b += part_b[i] * part_b[i];

        norm=sqrt(norm);
        norm_b=sqrt(norm_b);
        if (norm/norm_b <= eps) {
            flag = 0;

        }
            part_x[i] -= t * part_sum[i];

        }
        MPI_Allgather(part_x, number_of_elements, MPI_DOUBLE, x,
                   number_of_elements, MPI_DOUBLE,  MPI_COMM_WORLD);
    }
    MPI_Gather(part_x, number_of_elements, MPI_DOUBLE, x,
               number_of_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (process_rank == 0) {
        for (int i = 0; i < N-process_count+resN; ++i) {
            printf("%f \n", x[i]);
        }
        end_time = MPI_Wtime();
        printf("time taken - %f sec\n", end_time - start_time);
    }
    MPI_Finalize();
    return 0;
}
