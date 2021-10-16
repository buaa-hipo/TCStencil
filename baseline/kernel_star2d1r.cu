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
    __shared__ half data[16][16];
    __shared__ half halo[4][16];
    const int index = threadIdx.x + (threadIdx.y << 4);
    const int offset_base = (blockIdx.y + 1) * (N << 4) + ((blockIdx.x * tile_size + 1) << 8);
    #pragma unroll
    for (int iter = 0; iter < tile_size; iter++) {
        int offset = offset_base + (iter << 8);
        // data[threadIdx.y][threadIdx.x + 2] = A[offset + index - (N << 4) + 224];
        // data[threadIdx.y + 18][threadIdx.x + 2] = A[offset + index + (N << 4)];
        // data[threadIdx.x + 2][threadIdx.y] = A[offset + (threadIdx.x << 4) - 242 + threadIdx.y];
        // data[threadIdx.x + 2][threadIdx.y + 18] = A[offset + (threadIdx.x << 4) + 256 + threadIdx.y];
        halo[threadIdx.y][threadIdx.x] = A[offset + index2[index]];
        halo[threadIdx.y + 2][threadIdx.x] = A[offset + index2[index + 32]];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            data[threadIdx.y + (i << 1)][threadIdx.x] = A[(i << 5) + offset + index];
        }
        __syncthreads();
        if (threadIdx.x < 14) {
            #pragma unroll
            for (int i = 0; i < 7; i++) {
                int y = threadIdx.y + (i << 1);
                int x = threadIdx.x;
                C[offset + index + (i << 5) + 17] = data[y][x + 1] * __float2half(1) + data[y + 1][x + 1] * __float2half(-40) + data[y + 2][x + 1] * __float2half(7)
                                             + data[y + 1][x] * __float2half(3) + data[y + 1][x + 2] * __float2half(5);
            }
        }
        int x = threadIdx.x;
        if (threadIdx.y == 0) {
            if (x > 0 && x < 15) {
                C[offset + x] = halo[0][x] * __float2half(1) + data[0][x] * __float2half(-40) + data[1][x] * __float2half(7)
                              + data[0][x - 1] * __float2half(3) + data[0][x + 1] * __float2half(5);
                C[offset + x * 16] = data[x - 1][0] * __float2half(1) + data[x][0] * __float2half(-40) + data[x + 1][0] * __float2half(7)
                                   + halo[2][x] * __float2half(3) + data[x][1] * __float2half(5);
            } else if (x == 0) {
                C[offset] = halo[0][0] * __float2half(1) + data[0][0] * __float2half(-40) + data[1][0] * __float2half(7)
                          + halo[1][0] * __float2half(3) + data[0][1] * __float2half(5);
            } else {
                C[offset + 15] = halo[0][0] * __float2half(1) + data[0][0] * __float2half(-40) + data[1][0] * __float2half(7)
                               + data[14][0] * __float2half(3) + halo[3][0] * __float2half(5);
            }
        } else {
            if (x > 0 && x < 15) {
                C[offset + x + 240] = data[14][x] * __float2half(1) + data[15][x] * __float2half(-40) + halo[1][x] * __float2half(7)
                                    + data[15][x - 1] * __float2half(3) + data[15][x + 1] * __float2half(5);
                C[offset + x * 16 + 15] = data[x - 1][15] * __float2half(1) + data[x][15] * __float2half(-40) + data[x + 1][15] * __float2half(7)
                                        + data[x][14] * __float2half(3) + halo[3][x] * __float2half(5);
            } else if (x == 0) {
                C[offset + 240] = data[14][0] * __float2half(1) + data[15][0] * __float2half(-40) + halo[1][0] * __float2half(7)
                                + halo[2][15] * __float2half(3) + data[15][1] * __float2half(5);
            } else {
                C[offset + 255] = data[14][15] * __float2half(1) + data[15][15] * __float2half(-40) + halo[1][15] * __float2half(7)
                                + data[15][14] * __float2half(3) + halo[3][15] * __float2half(5);
            }
        }
    }
}