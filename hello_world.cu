#include "common/book.h"
__global__ void add(int a, int b, int* c)
{
    *c = a + b;
}

int main(void)
{
    int c;
    int* dev_c;
    cudaMalloc((void**) &dev_c, sizeof(int));
    add<<<1, 1>>>(2, 3, dev_c);
    cudaMemcpy(&c, &dev_c, sizeof(int), cudaMemcpyHostToDevice);
    printf("2+3=%d\n", c);
    cudaFree(dev_c);
    return(0);
}