#include <iostream>
#include <cmath>
#include <stdio.h>
#define BLOCK_SIZE 32

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
            setElement(m, i, j, (j == i) ? 1.0 : 0.0);
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

__device__ void printMatrix(const Matrix &m) {
    for(int i = 0; i < m.rows; ++i) {
        for(int j = 0; j < m.cols; ++j) {
            printf("%lf ", getElementDev(m, i, j));
        }
        printf("\n");
    }
}

__device__ int calculateRowStart(int n_rows, int threadNum) {
    int rows_per_process = n_rows / BLOCK_SIZE;
    int rem = n_rows % BLOCK_SIZE;
    return (threadNum <= rem) ? (rows_per_process + 1) * threadNum : (rows_per_process + 1) * rem + rows_per_process * (threadNum - rem);;
}

__device__ Matrix generateSubmatrix(Matrix &big_mat, int threadNum) {
    int rows_per_process = big_mat.rows / BLOCK_SIZE;
    int rem = big_mat.rows % BLOCK_SIZE;

    Matrix sub_mat;
    sub_mat.rows = (threadNum < rem) ? (rows_per_process + 1) : rows_per_process;
    sub_mat.cols = big_mat.cols;

    int row_start = calculateRowStart(big_mat.rows, threadNum);
    sub_mat.mat = &big_mat.mat[row_start * sub_mat.cols];

    return sub_mat;
}

__global__ void matrixInversion(Matrix m, Matrix res) { 
    Matrix local_mat = generateSubmatrix(m, threadIdx.x);

    extern __shared__ double pivot[];

    int thread_pivot_row_start = calculateRowStart(m.rows, threadIdx.x);
    int thread_pivot_row_end = thread_pivot_row_start + local_mat.rows;

    for (int i = 0; i < n; ++i) {
        if (i <= thread_pivot_row_start && i < thread_pivot_row_end) {
            double d = getElementDev(m, i - thread_pivot_row_start, i);

            for (int j = 0; j < i; ++j)
                pivot[j] = 0;

            for (int j = i; j < 2 * n; ++j) {
                double x = getElementDev(m, i - thread_pivot_row_start, j) / d;
                setElementDev(m, i - thread_pivot_row_start, j, x);
                pivot[j] = x;
            }
        }

        __syncthreads();
        for (int local_i = 0; local_i < m.rows; local_i++) {
            double factor = getElementDev(m, local_i, i);
            for (int j = i; j < 2 * n; ++j) {
                double x = getElementDev(m, local_i, j);
                setElementDev(m, i - thread_pivot_row_start, j, x - (pivot[j] * factor));
            }
        }
        __syncthreads();
    }

    for (int i = thread_pivot_row_start; i < thread_pivot_row_end; i++) {
        for (j = 0;j < res.cols;j++) {
            double x = getElementDev(local_mat, i - thread_pivot_row_start, j + m.rows);
            setElementDev(res, i, j, x);
        }
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

    Matrix d_res;
    d_res.rows = m.rows;
    d_res.cols = m.cols / 2;
    cudaMalloc((void**)&d_res.mat, size / 2);

    matrixInversion<<<1, BLOCK_SIZE>>>(d_m, d_res);

    Matrix res;
    res.rows = m.rows;
    res.cols = m.cols / 2;
    res.mat = new double[res.rows * res.cols];

    cudaMemcpy(res.mat, d_res.mat, size / 2, cudaMemcpyDeviceToHost);

    cudaFree(d_m.mat);
    cudaFree(d_res.mat);
    
    cudaDeviceSynchronize();

    return 0;
}