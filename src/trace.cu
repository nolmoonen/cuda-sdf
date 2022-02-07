#include <cuda_runtime.h>

#include "trace.h"

#include <stdio.h>

__global__ void hello()
{
    printf("hello\n");
}


void hello_cuda()
{
    hello<<<1, 1>>>();
}

