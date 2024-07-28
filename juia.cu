#include "common/book.h"
#include "common/cpu_bitmap.h"

#define DIM 500

struct cuComplex
{
    float r, i;
    __device__ cuComplex(float a, float b): r(a), i(b)
    {}

    __device__ float magnitude2(void) 
    {
        return r*r + i*i;
    }
    
    __device__ cuComplex operator+(const cuComplex&a)
    {
        return cuComplex(r+a.r, i+a.i );
    }

    __device__ cuComplex operator*(const cuComplex&a)
    {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i );
    }

};

__device__ 
int julia(int x, int y)
{
    float scale = 1.5;
    float jx = scale*(float)(DIM/2 - x)/(DIM/2);
    float jy = scale*(float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    for(int i = 0; i < 200; i++)
    {
        a = a*a + c;
        if (a.magnitude2() > 1000)
        {
            return 0;
        }
    }

    return 1;
}

__global__
void kernel(unsigned char* dev_bitmap)
{
    int x = blockIdx.x;
    int y = blockIdx.y;

    int offset = x+y*gridDim.x;

    int val = julia(x, y);
    dev_bitmap[offset*4+0] = 255*val;
    dev_bitmap[offset*4+1] = 0;
    dev_bitmap[offset*4+2] = 0;
    dev_bitmap[offset*4+3] = 255;
}
struct  DataBlock
{
    unsigned char* dev_bitmap;
};

int main(void)
{
    DataBlock data;
    CPUBitmap bitmap(DIM, DIM, &data);
    unsigned char* dev_bitmap;
    cudaMalloc((void**) &dev_bitmap, bitmap.image_size());
    data.dev_bitmap = dev_bitmap;
    clock_t start, stop;
    dim3 grid(DIM, DIM);
    start = clock();
    kernel<<<grid, 1>>>(dev_bitmap);
    stop = clock();
    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    cudaFree(dev_bitmap);
    float time = (float) (stop - start)/
                (float) CLOCKS_PER_SEC*1000.0f;
    printf("Time: %3.1f ms\n", time);
    bitmap.display_and_exit();
}