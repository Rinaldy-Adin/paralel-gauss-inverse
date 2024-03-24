#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void printMatrix(double *mat, int n) {
    printf("%d\n", n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", mat[i * n + j]);
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
        if (*n % size != 0) {
            printf("Number of rows must be divisible by the number of processes.\n");
            fclose(file);
            MPI_Finalize();
            exit(1);
        }

        *mat = (double *)malloc((*n) * (*n) * sizeof(double));
        for (int i = 0; i < (*n) * (*n); i++) {
            fscanf(file, "%lf", &(*mat)[i]);
        }
        fclose(file);
    }
}

void matrix_inversion(double *mat, int n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int start = rank * n / size;
    int end = (rank + 1) * n / size;

    double *identity = (double *)malloc(n * n * sizeof(double));
    for (int i = start; i < end; i++) {
        for (int j = 0; j < n; j++) {
            identity[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gauss-Jordan
    for (int i = 0; i < n; i++) {
        double divisor = mat[i * n + i];
        for (int j = 0; j < n; j++) {
            mat[i * n + j] /= divisor;
            identity[i * n + j] /= divisor;
        }

        for (int j = 0; j < n; j++) {
            if (j != i) {
                double factor = mat[j * n + i];
                for (int k = 0; k < n; k++) {
                    mat[j * n + k] -= factor * mat[i * n + k];
                    identity[j * n + k] -= factor * identity[i * n + k];
                }
            }
        }
    }

    for (int i = 0; i < n * n; i++) {
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

    matrix_inversion(local_mat, sendcounts[rank] / n);

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