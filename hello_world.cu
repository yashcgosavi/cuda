#include "common/book.h"

int main( void ) 
{
    cudaDeviceProp prop;
    int dev;

    HANDLE_ERROR(cudaGetDevice(&dev));
    printf("Id of current device: %d\n", dev);
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    cudaChooseDevice(&dev, &prop);
    printf("choosen device id: %d\n", dev);
    cudaSetDevice(dev);
}