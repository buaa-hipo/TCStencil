#include "device_launch_parameters.h"
#include "stdio.h"
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <vector>
#include <cooperative_groups.h>
#include "../common/common.hpp"

using namespace nvcuda;

extern "C" __global__ void kernel_run(half *__restrict__ A, half *__restrict__ C, int N, int tile_size, int *index1, int *index2) {
    __shared__ half data[20][16];
    __shared__ half halo[18 * 2];
    const int index = threadIdx.x + (threadIdx.y << 4);
    const int offset_base = (blockIdx.y + 1) * (N << 4) + ((blockIdx.x * tile_size + 1) << 8);
    #pragma unroll
    for (int iter = 0; iter < tile_size; iter++) {
        int offset = offset_base + (iter << 8);
        data[threadIdx.y][threadIdx.x] = A[offset + index - (N << 4) + 224];
        data[threadIdx.y + 18][threadIdx.x] = A[offset + index + (N << 4)];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            data[threadIdx.y + (i << 1) + 2][threadIdx.x] = A[(i << 5) + offset + index];
        }
        // load halo
        halo[threadIdx.y * 18 + threadIdx.x + 1] =  A[(threadIdx.x << 4) + offset - 241 + threadIdx.y * 497];
        if (index < 4) {
            halo[index1[index]] = A[offset + index2[index]];
        }
        __syncthreads();
        if (threadIdx.x > 0 && threadIdx.x < 15) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int y = threadIdx.y + (i << 1);
                int x = threadIdx.x - 1;
                C[offset + index + (i << 5)] = data[y][x] * __float2half(0) + data[y][x + 1] * __float2half(1) + data[y][x + 2] * __float2half(2) 
                                            + data[y + 1][x] * __float2half(3) + data[y + 1][x + 1] * __float2half(-40) + data[y + 1][x + 2] * __float2half(5) 
                                            + data[y + 2][x] * __float2half(6) + data[y + 2][x + 1] * __float2half(7) + data[y + 2][x + 2] * __float2half(8);
            }
        }
        if (threadIdx.y == 0) {
            int x = threadIdx.x;
            C[offset + x * 16] = data[x][0] * __float2half(1) + data[x][1] * __float2half(2) + data[x + 1][0] * __float2half(-40)
                               + data[x + 1][1] * __float2half(5) + data[x + 2][0] * __float2half(7) + data[x + 2][0] * __float2half(8)
                               + halo[x] * __float2half(0) + halo[x + 1] * __float2half(3) + halo[x + 2] * __float2half(6);
        } else {
            int x = threadIdx.x;
            int base = x + 18;
            C[offset + x * 16 + 15] = data[x][14] * __float2half(0) + data[x][15] * __float2half(1) + data[x + 1][14] * __float2half(3) 
                                    + data[x + 1][15] * __float2half(-40) + data[x + 2][14] * __float2half(6) + data[x + 2][15] * __float2half(7)
                                    + halo[base] * __float2half(2) + halo[base + 1] * __float2half(5) + halo[base + 2] * __float2half(8);
        }
    }
}