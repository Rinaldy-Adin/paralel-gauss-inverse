// gcc mp.c --openmp -o mp

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    #pragma omp parallel num_threads(8) 
    {
        int nthreads, tid;
        nthreads = omp_get_num_threads();
        tid = omp_get_thread_num();
        printf("Hello world from from thread %d out of %d threads\n", tid, nthreads);
    }
    return 0;
}