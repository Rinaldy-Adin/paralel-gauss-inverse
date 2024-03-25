#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void printMatrix(double *mat, int n) {
    printf("%d\n", n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%lf ", mat[i * n + j]);
        }
        printf("\n");
    }
}

void readMatrixFromFile(char *filename, double **mat, int *n, int rank, int size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        if (rank == 0) {
            printf("Error opening file %s\n", filename);
        }
        MPI_Finalize();
        exit(1);
    }

    if (rank == 0) {
        fscanf(file, "%d", n);

        *mat = (double *)malloc((*n) * (*n) * sizeof(double));
        for (int i = 0; i < (*n) * (*n); i++) {
            fscanf(file, "%lf", &(*mat)[i]);
        }
        fclose(file);
    }
}

void matrix_inversion(double *mat, int n, int local_n, int start_row, int end_row) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *identity = (double *)malloc(local_n * n * sizeof(double));
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            identity[(i - start_row) * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gauss-Jordan
    double *pivot_mat = (double *)malloc(n * sizeof(double));
    double *pivot_ident = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        int pivot_process = i * size / n;
        int local_pivot = i - start_row;

        if (rank == pivot_process) {
            double divisor = mat[local_pivot * n + i];
            for (int j = 0; j < n; j++) {
                mat[local_pivot * n + j] /= divisor;
                identity[local_pivot * n + j] /= divisor;

                pivot_mat[j] = mat[local_pivot * n + j];
                pivot_ident[j] = identity[local_pivot * n + j];
            }
        }

        MPI_Bcast(pivot_mat, n, MPI_DOUBLE, pivot_process, MPI_COMM_WORLD);
        MPI_Bcast(pivot_ident, n, MPI_DOUBLE, pivot_process, MPI_COMM_WORLD);

        for (int local_i = 0; local_i < local_n; local_i++) {
            if (!(rank == pivot_process && local_i == local_pivot)) {
                double factor = mat[local_i * n + i];
                for (int j = 0; j < n; j++) {
                    mat[local_i * n + j] -= factor * pivot_mat[j];
                    identity[local_i * n + j] -= factor * pivot_ident[j];
                }
            }
        }
    }

    for (int i = 0; i < local_n * n; i++) {
        mat[i] = identity[i];
    }

    free(identity);
}

void performMatrixOperations(int n, int size, int rank, double *mat) {
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_process = n / size;
    int remainder = n % size;

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    int displacement = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (i < remainder) ? (rows_per_process + 1) * n : rows_per_process * n;
        displs[i] = displacement;
        displacement += sendcounts[i];
    }

    double *local_mat = (double *)malloc(sendcounts[rank] * sizeof(double));
    MPI_Scatterv(mat, sendcounts, displs, MPI_DOUBLE, local_mat, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix_inversion(local_mat, n, sendcounts[rank] / n, displs[rank] / n, (displs[rank] + sendcounts[rank]) / n);

    double *result_mat = NULL;
    if (rank == 0) {
        result_mat = (double *)malloc(n * n * sizeof(double));
    }
    MPI_Gatherv(local_mat, sendcounts[rank], MPI_DOUBLE, result_mat, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printMatrix(result_mat, n);
        free(result_mat);
    }

    free(local_mat);
    free(sendcounts);
    free(displs);
}


int main(int argc, char *argv[]) {
    int rank, size, n;
    double *mat;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <input_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    char *filename = argv[1];
    readMatrixFromFile(filename, &mat, &n, rank, size);

    performMatrixOperations(n, size, rank, mat);
    MPI_Finalize();
    return 0;
}