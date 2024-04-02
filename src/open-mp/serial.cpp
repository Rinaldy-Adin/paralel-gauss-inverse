#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;

struct Matrix {
    int size;
    double **mat;
};

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

void freeMatrix(Matrix &m) {
    for(int i = 0; i < m.size; ++i)
        delete[] m.mat[i];
    delete[] m.mat; 
}

void readMatrix(Matrix &m) {
    int n;
    cin >> n;
    allocateMatrix(m, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            cin >> m.mat[i][j];
}

void gaussJordan(Matrix &m) {
    int n = m.size;

    for(int i = 0; i < n; ++i) {
        double d = m.mat[i][i];
        for(int j = 0; j < 2*n; ++j) {
            m.mat[i][j] /= d;
        }

        for(int k = 0; k < n; ++k) {
            if(k != i) {
                double factor = m.mat[k][i];
                for(int j = 0; j < 2*n; ++j) {
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
    gaussJordan(m);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;
    
    printMatrix(m);
    cout << "Time taken by function: " << duration.count() << " seconds" << endl;


    freeMatrix(m);

    return 0;
}
