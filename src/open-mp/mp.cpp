#include <iostream>
#include <chrono>
#include <omp.h>
#include <cmath>

using namespace std;

struct Matrix {
    int size;
    double **mat;
};

// Read Matrix
void allocateMatrix(Matrix &m, int size) {
    m.size = size;
    m.mat = new double*[size]; 
    for(int i = 0; i < size; ++i) {
        m.mat[i] = new double[2*size];
        for(int j = 0; j < 2*size; ++j) {
            m.mat[i][j] = (j - size == i) ? 1.0 : 0.0;
        }
    }
}

void readMatrix(Matrix &m) {
    int n;
    cin >> n;
    allocateMatrix(m, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            cin >> m.mat[i][j];
}

void freeMatrix(Matrix &m) {
    for(int i = 0; i < m.size; ++i)
        delete[] m.mat[i];
    delete[] m.mat; 
}

// Algorithm
void partialPivoting(Matrix &m, int pivot) {
    int maxIndex = pivot;
    double maxVal = 0.0; 
    int n = m.size;
    #pragma omp parallel
    {
        double localMaxVal = 0.0;
        int localMaxIndex = pivot;
        // Distribute loop
        #pragma omp for nowait
        for (int i = pivot; i < n; i++) {
            double val = std::abs(m.mat[i][pivot]);
            if (val > localMaxVal) {
                localMaxVal = val;
                localMaxIndex = i;
            }
        }
        #pragma omp critical
        {
            if (localMaxVal > maxVal) {
                maxVal = localMaxVal;
                maxIndex = localMaxIndex;
            }
        }
    }

    if (maxIndex != pivot) {
        swap(m.mat[pivot], m.mat[maxIndex]);
    }
}

void matrixInversion(Matrix &m) {
    int n = m.size;

    for (int i = 0; i < n; ++i) {
        partialPivoting(m, i);

        double d = m.mat[i][i];
        for (int j = 0; j < 2 * n; ++j) {
            m.mat[i][j] /= d;
        }

        // Paralelisasi di bagian eliminasi
        #pragma omp parallel for
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = m.mat[k][i];
                for (int j = 0; j < 2 * n; ++j) {
                    #pragma omp atomic
                    m.mat[k][j] -= factor * m.mat[i][j];
                }
            }
        }
    }
}


void printMatrix(const Matrix &m) {
    int n = m.size;
    for(int i = 0; i < n; ++i) {
        for(int j = n; j < 2*n; ++j) {
            cout << m.mat[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    Matrix m;
    
    readMatrix(m);
    auto start = std::chrono::high_resolution_clock::now();
    matrixInversion(m);

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;
    

    printMatrix(m);
    cout << "Time taken by function: " << duration.count() << " seconds" << endl;


    freeMatrix(m);

    return 0;
}
