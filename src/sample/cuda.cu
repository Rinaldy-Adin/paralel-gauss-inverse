#include <stdio.h>

__global__ void device_hello_world() {
    printf("Hello world from x.%d y.%d z.%d!\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(void) {
    dim3 block(2, 2, 2);    // x*y*z <= 1024
    dim3 grid(1, 1, 1);
    for (int i = 0; i < 4; i++)
        device_hello_world<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}