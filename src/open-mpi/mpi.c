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

int readMatrixFromFile(char *filename, double **mat, int *n, int rank, int size) {
    int ok = 1;
    FILE *file;
    if (rank == 0) {
        file = fopen(filename, "r");
        if (!file) {
            printf("Error opening file %s\n", filename);
            ok = 0;
        }
    }

    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) {
        return 1;
    }

    if (rank == 0) {
        fscanf(file, "%d", n);

        *mat = (double *)malloc((*n) * (*n) * sizeof(double));
        for (int i = 0; i < (*n) * (*n); i++) {
            fscanf(file, "%lf", &(*mat)[i]);
        }
        fclose(file);
    }
    return 0;
}

void partialPivoting(double *mat, double *ident, int n) {
    double d;
    for(int i = n; i > 1; i--) {
        if(mat[(i-1) * n + 1] < mat[i * n + 1]) {
            for(int j = 0; j < n; j++) {
                d = mat[i * n  + j];
                mat[i * n  + j] = mat[(i-1) * n + j];
                mat[(i-1) * n + j] = d;

                d = ident[i * n  + j];
                ident[i * n  + j] = ident[(i-1) * n + j];
                ident[(i-1) * n + j] = d;
            }
        }
    }
}

void generateIdent(double *ident, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ident[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void matrix_inversion(double *mat, double *identity, int n, int local_n, int start_row, int end_row) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
}

void performMatrixOperations(int n, int size, int rank, double *mat, double *ident) {
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
    double *local_ident = (double *)malloc(sendcounts[rank] * sizeof(double));
    MPI_Scatterv(mat, sendcounts, displs, MPI_DOUBLE, local_mat, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(ident, sendcounts, displs, MPI_DOUBLE, local_ident, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix_inversion(local_mat, local_ident, n, sendcounts[rank] / n, displs[rank] / n, (displs[rank] + sendcounts[rank]) / n);

    double *result_mat = NULL;
    if (rank == 0) {
        result_mat = (double *)malloc(n * n * sizeof(double));
    }
    MPI_Gatherv(local_ident, sendcounts[rank], MPI_DOUBLE, result_mat, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printMatrix(result_mat, n);
        free(result_mat);
    }

    free(local_mat);
    free(local_ident);
    free(sendcounts);
    free(displs);
}


int main(int argc, char *argv[]) {
    int rank, size, n;
    double *mat, *ident;

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
    int fail_read = readMatrixFromFile(filename, &mat, &n, rank, size);

    if (!fail_read) {
        if (rank == 0) {
            ident = (double *)malloc(n * n * sizeof(double));
            generateIdent(ident, n);
            partialPivoting(mat, ident, n);
        }

        performMatrixOperations(n, size, rank, mat, ident);

        if (rank == 0) {
            free(mat);
            free(ident);
        }
    }
    MPI_Finalize();

    return 0;
}