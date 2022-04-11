#include <iostream>
#include <mpi.h>

#define N1 2000
#define N2 2000
#define N3 2000
#define DIM_NUM 2


void matrices_init(double *A, double *B) {
    for (int i = 0; i < N1; ++i) {
        for (int j = 0; j < N2; ++j) {
            A[i * N1 + j] = 1.0;
        }
    }
    for (int i = 0; i < N2; ++i) {
        for (int j = 0; j < N3; ++j) {
            B[i * N2 + j] = 1.0;
        }
    }
}


void create_type(int *mtx_dims, int *block_size, MPI_Datatype *b_type, MPI_Datatype *c_type) {
    MPI_Datatype col1, col2;

    MPI_Type_vector(mtx_dims[1], block_size[1], mtx_dims[2], MPI_DOUBLE, &col1);
    MPI_Type_commit(&col1);
    MPI_Type_create_resized(col1, 0, sizeof(double) * block_size[1], b_type);
    MPI_Type_commit(b_type);

    MPI_Type_vector(block_size[0], block_size[1], mtx_dims[2], MPI_DOUBLE, &col2);
    MPI_Type_commit(&col2);
    MPI_Type_create_resized(col1, 0, sizeof(double) * block_size[1], c_type);
    MPI_Type_commit(c_type);

    MPI_Type_free(&col1);
    MPI_Type_free(&col2);
}

void calculate_submatrix(int *b_disp, int *b_count,
                         int *c_disp, int *c_count, const int *web_conf,
                         const int *block_size) {
    for (int i = 0; i < web_conf[1]; ++i) {
        b_disp[i] = i;
        b_count[i] = 1;
    }
    for (int i = 0; i < web_conf[0]; ++i) {
        for (int j = 0; j < web_conf[1]; ++j) {
            c_disp[i * web_conf[1] + j] = i * web_conf[1] *
                                          block_size[0] + j;
            c_count[i * web_conf[1] + j] = 1;
        }
    }
}

void calculate(double *A, double *B, double *C, int *mtx_dims,
               int *web_conf, MPI_Comm comm) {
    MPI_Comm comm_copy;
    MPI_Comm_dup(comm, &comm_copy);

    MPI_Bcast(mtx_dims, 3, MPI_INT, 0, comm_copy);
    MPI_Bcast(web_conf, 2, MPI_INT, 0, comm_copy);
    int periods[2] = { 0 };
    MPI_Comm comm_2D;
    MPI_Cart_create(comm_copy, DIM_NUM, web_conf, periods, 0,
                    &comm_2D);
    int rank;
    MPI_Comm_rank(comm_2D, &rank);

    int coords[2];
    MPI_Cart_coords(comm_2D, rank, DIM_NUM, coords);
    MPI_Comm comm_1D[2];
    int remain_dims[2];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            remain_dims[j] = (i == j);
        }
        MPI_Cart_sub(comm_2D, remain_dims, &comm_1D[i]);
    }

    int block_size[2];
    block_size[0] = mtx_dims[0] / web_conf[0];
    block_size[1] = mtx_dims[2] / web_conf[1];

    double *A_stripe = new double[block_size[0] * mtx_dims[1]];
    double *B_column = new double[mtx_dims[1] * block_size[1]];

    double *C_block = new double[block_size[0] * block_size[1]];

    MPI_Datatype b_type, c_type;
    int *b_disp, *b_count, *c_disp, *c_count;

    if (rank == 0) {
        create_type(mtx_dims, block_size, &b_type, &c_type);
        b_disp = new int [web_conf[1]];
        b_count = new int [web_conf[1]];
        c_disp = new int [web_conf[0] * web_conf[1]];
        c_count = new int [web_conf[0] * web_conf[1]];
        calculate_submatrix(b_disp, b_count, c_disp, c_count,web_conf, block_size);
    }

    if (coords[1] == 0) {
        MPI_Scatter(A, block_size[0] * mtx_dims[1], MPI_DOUBLE,
                    A_stripe, block_size[0] * mtx_dims[1], MPI_DOUBLE, 0,
                    comm_1D[0]);
    }

    if (coords[0] == 0) {
        MPI_Scatterv(B, b_count, b_disp, b_type, B_column,
                     mtx_dims[1] * block_size[1], MPI_DOUBLE, 0, comm_1D[1]);
    }
    MPI_Bcast(A_stripe, block_size[0] * mtx_dims[1], MPI_DOUBLE, 0, comm_1D[1]);


    MPI_Bcast(B_column, mtx_dims[1] * block_size[1], MPI_DOUBLE, 0, comm_1D[0]);
    
    for (int i = 0; i < block_size[0]; ++i) {
        for (int j = 0; j < block_size[1]; ++j) {
            for (int k = 0; k < mtx_dims[1]; ++k) {
                C_block[i * block_size[1] + j] += A_stripe[i * mtx_dims[1] + k] * B_column[k * block_size[1] + j];
            }
        }
    }

    MPI_Gatherv(C_block, block_size[0] * block_size[1], MPI_DOUBLE, C, c_count, c_disp, c_type, 0, comm_2D);

    delete[] A_stripe;
    delete[] B_column;
    delete[] C_block;

    MPI_Comm_free(&comm_copy);
    MPI_Comm_free(&comm_2D);
    for (int i = 0; i < 2; ++i) {
        MPI_Comm_free(&comm_1D[i]);
    }
    if (rank == 0) {
        delete[] b_disp;
        delete[] b_count;
        delete[] c_disp;
        delete[] c_count;
        MPI_Type_free(&b_type);
        MPI_Type_free(&c_type);

    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dims[DIM_NUM] = {0, 0};
    MPI_Dims_create(size, DIM_NUM, dims);
    int mtx_dims[3];

    std::cout << std::endl<< "Dims: " << dims[0] << "x" << dims[1];

    double *A, *B, *C;

    if (rank == 0) {
        A = new double[N1 * N2];
        B = new double[N2 * N3];
        C = new double[N1 * N3];

        matrices_init(A, B);

        mtx_dims[0] = N1;
        mtx_dims[1] = N2;
        mtx_dims[2] = N3;
    }

    double start = MPI_Wtime();
    calculate(A, B, C, mtx_dims, dims, MPI_COMM_WORLD);
    double finish = MPI_Wtime();

    if (rank == 0) {
        std::cout<< std::endl << "Time: " << (finish - start) << std::endl;
        delete[] A;
        delete[] B;
        delete[] C;
    }

    MPI_Finalize();
    return 0;
}

