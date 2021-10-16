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
    __shared__ half halo[80];
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
        halo[threadIdx.y * 20 + threadIdx.x + 2] =  A[(threadIdx.x << 4) + offset - 242 + threadIdx.y];
        halo[threadIdx.y * 20 + threadIdx.x + 42] = A[(threadIdx.x << 4) + offset + 256 + threadIdx.y];
        if (threadIdx.y == 0) {
            halo[index1[index]] = A[offset + index2[index]];
        }
        __syncthreads();
        if (threadIdx.x > 1 && threadIdx.x < 14) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int y = threadIdx.y + (i << 1);
                int x = threadIdx.x - 2;
                C[offset + index + (i << 5)] = data[y][x] * __float2half(0.0) + data[y][x + 1] * __float2half(1.0) + data[y][x + 2] * __float2half(2.0) 
                                            + data[y][x + 3] * __float2half(3.0) + data[y][x + 4] * __float2half(4.0)
                                            + data[y + 1][x] * __float2half(5.0) + data[y + 1][x + 1] * __float2half(6.0) + data[y + 1][x + 2] * __float2half(7.0) 
                                            + data[y + 1][x + 3] * __float2half(8.0) + data[y + 1][x + 4] * __float2half(9.0)
                                            + data[y + 2][x] * __float2half(10.0) + data[y + 2][x + 1] * __float2half(11.0) + data[y + 2][x + 2] * __float2half(-290) 
                                            + data[y + 2][x + 3] * __float2half(13.0) + data[y + 2][x + 4] * __float2half(14.0)
                                            + data[y + 3][x] * __float2half(15.0) + data[y + 3][x + 1] * __float2half(16.0) + data[y + 3][x + 2] * __float2half(17.0) 
                                            + data[y + 3][x + 3] * __float2half(18.0) + data[y + 3][x + 4] * __float2half(19.0)
                                            + data[y + 4][x] * __float2half(20.0) + data[y + 4][x + 1] * __float2half(21.0) + data[y + 4][x + 2] * __float2half(22.0) 
                                            + data[y + 4][x + 3] * __float2half(23.0) + data[y + 4][x + 4] * __float2half(24.0);
            }
        }
        if (threadIdx.y == 0) {
            int x = threadIdx.x;
            C[offset + threadIdx.x * 16] = data[x][0] * __float2half(2.0) + data[x][1] * __float2half(3.0) + data[x][2] * __float2half(4.0)
                                        + data[x + 1][0] * __float2half(7.0) + data[x + 1][1] * __float2half(8.0) + data[x + 1][2] * __float2half(9.0)
                                        + data[x + 2][0] * __float2half(-290) + data[x + 2][1] * __float2half(13.0) + data[x + 2][2] * __float2half(14.0)
                                        + data[x + 3][0] * __float2half(17.0) + data[x + 3][1] * __float2half(18.0) + data[x + 3][2] * __float2half(19.0)
                                        + data[x + 4][0] * __float2half(22.0) + data[x + 4][1] * __float2half(23.0) + data[x + 4][2] * __float2half(24.0)
                                        + halo[x] * __float2half(0) + halo[x + 1] * __float2half(5) + halo[x + 2] * __float2half(10)
                                        + halo[x + 3] * __float2half(15) + halo[x + 4] * __float2half(20)
                                        + halo[x + 20] * __float2half(1) + halo[x + 21] * __float2half(6) + halo[x + 22] * __float2half(11)
                                        + halo[x + 23] * __float2half(16) + halo[x + 24] * __float2half(21);
            C[offset + threadIdx.x * 16 + 1] = data[x][0] * __float2half(1) + data[x][1] * __float2half(2) + data[x][2] * __float2half(3)
                                            + data[x][3] * __float2half(4)
                                            + data[x + 1][0] * __float2half(6) + data[x + 1][1] * __float2half(7) + data[x + 1][2] * __float2half(8)
                                            + data[x + 1][3] * __float2half(9)
                                            + data[x + 2][0] * __float2half(11) + data[x + 2][1] * __float2half(-290) + data[x + 2][2] * __float2half(13)
                                            + data[x + 2][3] * __float2half(14)
                                            + data[x + 3][0] * __float2half(16) + data[x + 3][1] * __float2half(17) + data[x + 3][2] * __float2half(18)
                                            + data[x + 3][3] * __float2half(19)
                                            + data[x + 4][0] * __float2half(21) + data[x + 4][1] * __float2half(22) + data[x + 4][2] * __float2half(23)
                                            + data[x + 4][3] * __float2half(24)
                                            + halo[x + 20] * __float2half(0) + halo[x + 21] * __float2half(5) + halo[x + 22] * __float2half(10) +
                                            + halo[x + 23] * __float2half(15) + halo[x + 24] * __float2half(20);
        } else {
            int x = threadIdx.x;
            int base = x + 40;
            C[offset + x * 16 + 15] = data[x][13] * __float2half(0) + data[x][14] * __float2half(1) + data[x][15] * __float2half(2)
                                   + data[x + 1][13] * __float2half(5) + data[x + 1][14] * __float2half(6) + data[x + 1][15] * __float2half(7)
                                   + data[x + 2][13] * __float2half(10) + data[x + 2][14] * __float2half(11) + data[x + 2][15] * __float2half(-290)
                                   + data[x + 3][13] * __float2half(15) + data[x + 3][14] * __float2half(16) + data[x + 3][15] * __float2half(17)
                                   + data[x + 4][13] * __float2half(20) + data[x + 4][14] * __float2half(21) + data[x + 4][15] * __float2half(22)
                                   + halo[base] * __float2half(3) + halo[base + 1] * __float2half(8) + halo[base + 2] * __float2half(13)
                                   + halo[base + 3] * __float2half(18) + halo[base + 4] * __float2half(23)
                                   + halo[base + 20] * __float2half(4) + halo[base + 21] * __float2half(9) + halo[base + 22] * __float2half(14)
                                   + halo[base + 23] * __float2half(19) + halo[base + 24] * __float2half(24);
            C[offset + x * 16 + 14] = data[x][12] * __float2half(0) + data[x][13] * __float2half(1) + data[x][14] * __float2half(2)
                                   + data[x][15] * __float2half(3)
                                   + data[x + 1][12] * __float2half(5) + data[x + 1][13] * __float2half(6) + data[x + 1][14] * __float2half(7)
                                   + data[x + 1][15] * __float2half(8)
                                   + data[x + 2][12] * __float2half(10) + data[x + 2][13] * __float2half(11) + data[x + 2][14] * __float2half(-290)
                                   + data[x + 2][15] * __float2half(13)
                                   + data[x + 3][12] * __float2half(15) + data[x + 3][13] * __float2half(16) + data[x + 3][14] * __float2half(17)
                                   + data[x + 3][15] * __float2half(18)
                                   + data[x + 4][12] * __float2half(20) + data[x + 4][13] * __float2half(21) + data[x + 4][14] * __float2half(22)
                                   + data[x + 4][15] * __float2half(23)
                                   + halo[base] * __float2half(4) + halo[base + 1] * __float2half(9) + halo[base + 2] * __float2half(14)
                                   + halo[base + 3] * __float2half(19) + halo[base + 4] * __float2half(24);
        }
    }
}