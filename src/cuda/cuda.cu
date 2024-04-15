#include <iostream>
#include <cmath>
#include <stdio.h>

using namespace std;

struct Matrix {
    int rows;
    int cols;
    double *mat;
};

double getElement(Matrix m, int row, int col) {
    return m.mat[row * m.cols + col];
}

void setElement(Matrix m, int row, int col, double val) {
    m.mat[row * m.cols + col] = val;
}

__device__ double getElementDev(Matrix m, int row, int col) {
    return m.mat[row * m.cols + col];
}

__device__ void setElementDev(Matrix m, int row, int col, double val) {
    m.mat[row * m.cols + col] = val;
}

void allocateMatrix(Matrix &m, int size) {
    m.rows = size;
    m.cols = size * 2;
    m.mat = new double[m.rows * m.cols];
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < 2*size; ++j) {
            setElement(m, i, j, (j - size == i) ? 1.0 : 0.0);
        }
    }
}

void readMatrix(Matrix &m) {
    int n;
    cin >> n;
    allocateMatrix(m, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j){
            double x;
            cin >> x;
            setElement(m, i, j, x);
        }
}

void printMatrix(const Matrix &m) {
    for(int i = 0; i < m.rows; ++i) {
        for(int j = 0; j < m.cols; ++j) {
            printf("%lf ", getElement(m, i, j));
        }
        printf("\n");
    }
}

void printIdent(const Matrix &m) {
    for(int i = 0; i < m.rows; ++i) {
        for(int j = m.rows; j < m.cols; ++j) {
            printf("%lf ", getElement(m, i, j));
        }
        printf("\n");
    }
}

__device__ void printMatrixDev(const Matrix &m) {
    for(int i = 0; i < m.rows; ++i) {
        for(int j = 0; j < m.cols; ++j) {
            printf("%lf ", getElementDev(m, i, j));
        }
        printf("\n");
    }
}

__device__ int calculateRowStart(int n_rows, int threadNum, int blockSize) {
    int rows_per_process = n_rows / blockSize;
    int rem = n_rows % blockSize;
    return (threadNum <= rem) ? (rows_per_process + 1) * threadNum : (rows_per_process + 1) * rem + rows_per_process * (threadNum - rem);;
}

__device__ Matrix generateSubmatrix(Matrix &big_mat, int threadNum, int blockSize) {
    int rows_per_process = big_mat.rows / blockSize;
    int rem = big_mat.rows % blockSize;

    Matrix sub_mat;
    sub_mat.rows = (threadNum < rem) ? (rows_per_process + 1) : rows_per_process;
    sub_mat.cols = big_mat.cols;

    int row_start = calculateRowStart(big_mat.rows, threadNum, blockSize);
    sub_mat.mat = &big_mat.mat[row_start * sub_mat.cols];

    return sub_mat;
}

__global__ void matrixInversion(Matrix m, int blockSize) { 
    Matrix local_mat = generateSubmatrix(m, threadIdx.x, blockSize);

    extern __shared__ double pivot[];

    int thread_pivot_row_start = calculateRowStart(m.rows, threadIdx.x, blockSize);
    int thread_pivot_row_end = thread_pivot_row_start + local_mat.rows;
    int n = m.rows;

    if (threadIdx.x == 1) {
        
    }

    for (int i = 0; i < n; ++i) {
        if (thread_pivot_row_start <= i && i < thread_pivot_row_end) {
            double d = getElementDev(local_mat, i - thread_pivot_row_start, i);

            for (int j = 0; j < 2 * n; ++j) {
                double x = getElementDev(local_mat, i - thread_pivot_row_start, j) / d;
                setElementDev(local_mat, i - thread_pivot_row_start, j, x);
                pivot[j] = x;
                // printf("%lf ", x);
            }
            // printf("\n");
        }

        __syncthreads();
        for (int local_i = 0; local_i < local_mat.rows; local_i++) {
            if (thread_pivot_row_start <= i && i < thread_pivot_row_end && local_i == i - thread_pivot_row_start)
                continue;

            double factor = getElementDev(local_mat, local_i, i);

            if (threadIdx.x == 1) {
                // printf("i %d\n", i);
                // printf("factor %lf\n", factor);
            }

            for (int j = 0; j < 2 * n; ++j) {
                double x = getElementDev(local_mat, local_i, j);
                double y = x - (pivot[j] * factor);
                setElementDev(local_mat, local_i, j, y);
                if (threadIdx.x == 1) {
                    // printf("%lf ", y);
                }
            }
            if (threadIdx.x == 1) {
                // printf("\n\n");
            }
        }
        __syncthreads();
    }
}

int main(void) {
    Matrix m;
    readMatrix(m);

    size_t size = m.rows * m.cols * sizeof(double);
    Matrix d_m;
    d_m.rows = m.rows;
    d_m.cols = m.cols;
    cudaMalloc((void**)&d_m.mat, size);
    cudaMemcpy(d_m.mat, m.mat, size, cudaMemcpyHostToDevice);    

    int blockSize = m.rows;
    int sharedMemSize = blockSize * sizeof(double) * 2;

    matrixInversion<<<1, blockSize, sharedMemSize>>>(d_m, blockSize);

    cudaMemcpy(m.mat, d_m.mat, size, cudaMemcpyDeviceToHost);

    printf("%d\n", m.rows);
    printIdent(m);
    cudaFree(d_m.mat);
    
    cudaDeviceSynchronize();

    return 0;
}