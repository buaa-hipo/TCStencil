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
    __shared__ half halo[8][16];
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
        halo[threadIdx.y + 4][threadIdx.x] = A[offset + index2[index + 64]];
        halo[threadIdx.y + 6][threadIdx.x] = A[offset + index2[index + 96]];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            data[threadIdx.y + (i << 1)][threadIdx.x] = A[(i << 5) + offset + index];
        }
        __syncthreads();
        if (threadIdx.x < 12) {
            #pragma unroll
            for (int i = 0; i < 6; i++) {
                int y = threadIdx.y + (i << 1);
                int x = threadIdx.x;
                C[offset + index + (i << 5) + 34] = data[y][x + 2] * __float2half(2.0) + data[y + 1][x + 2] * __float2half(7.0) + data[y + 2][x + 2] * __float2half(-290.0)
                                             + data[y + 3][x + 2] * __float2half(17.0) + data[y + 4][x + 2] * __float2half(22.0) + data[y + 2][x] * __float2half(10.0)
                                             + data[y + 2][x + 1] * __float2half(11.0) + data[y + 2][x + 3] * __float2half(13.0) + data[y + 2][x + 4] * __float2half(14.0);
            }
        }
        int x = threadIdx.x;
        __shared__ half tmp[4][2][2];
        if (threadIdx.y == 0) {
            if (x > 1 && x < 14) {
                C[offset + x] = halo[0][x] * __float2half(2) + halo[1][x] * __float2half(7) + data[0][x] * __float2half(-290) + data[1][x] * __float2half(17)
                              + data[2][x] * __float2half(22) + data[0][x - 2] * __float2half(10) + data[0][x - 1] * __float2half(11)
                              + data[0][x + 1] * __float2half(13) + data[0][x + 2] * __float2half(14);
                C[offset + x + 16] = halo[1][x] * __float2half(2) + data[0][x] * __float2half(7) + data[1][x] * __float2half(-290) + data[2][x] * __float2half(17)
                                   + data[3][x] * __float2half(22) + data[1][x - 2] * __float2half(10) + data[1][x - 1] * __float2half(11)
                                   + data[1][x + 1] * __float2half(13) + data[1][x + 2] * __float2half(14);
                C[offset + x * 16] = data[x - 2][0] * __float2half(2) + data[x - 2][0] * __float2half(7) + data[x][0] * __float2half(-290)
                                   + data[x + 1][0] * __float2half(17) + data[x + 2][0] * __float2half(22) + halo[4][x] * __float2half(10)
                                   + halo[5][x] * __float2half(11) + data[x][1] * __float2half(13) + data[x][2] * __float2half(14);
                C[offset + x * 16 + 1] = data[x - 2][1] * __float2half(2) + data[x - 2][1] * __float2half(7) + data[x][1] * __float2half(-290)
                                       + data[x + 1][1] * __float2half(17) + data[x + 2][1] * __float2half(22) + halo[5][x] * __float2half(10)
                                       + data[x][0] * __float2half(11) + data[x][2] * __float2half(13) + data[x][3] * __float2half(14);
            } else if (x < 2) {
                tmp[0][0][x] = halo[0][x] * __float2half(2) + halo[1][x] * __float2half(7) + data[0][x] * __float2half(-290) + data[1][x] * __float2half(17)
                             + data[2][x] * __float2half(22);
                tmp[0][1][x] = halo[1][x] * __float2half(2) + data[0][x] * __float2half(7) + data[1][x] * __float2half(-290) + data[2][x] * __float2half(17)
                             + data[3][x] * __float2half(22);
                C[offset + x * 16] = tmp[0][x][0] + halo[4][x] * __float2half(10) + halo[5][x] * __float2half(11) + data[x][1] * __float2half(13)
                                   + data[x][2] * __float2half(14);
                C[offset + x * 16 + 1] = tmp[0][x][1] + halo[5][x] * __float2half(10) + data[x][0] * __float2half(11) + data[x][2] * __float2half(13)
                                       + data[x][3] * __float2half(14);
            } else {
                tmp[1][0][x - 14] = halo[0][x] * __float2half(2) + halo[1][x] * __float2half(7) + data[0][x] * __float2half(-290) + data[1][x] * __float2half(17)
                                  + data[2][x] * __float2half(22);
                tmp[1][1][x - 14] = halo[1][x] * __float2half(2) + data[0][x] * __float2half(7) + data[1][x] * __float2half(-290) + data[2][x] * __float2half(17)
                                  + data[3][x] * __float2half(22);
                C[offset + x * 16] = tmp[2][x - 14][0] + halo[4][x] * __float2half(10) + halo[5][x] * __float2half(11) + data[x][1] * __float2half(13)
                                   + data[x][2] * __float2half(14);
                C[offset + x * 16 + 1] = tmp[2][x - 14][1] + halo[5][x] * __float2half(10) + data[x][0] * __float2half(11) + data[x][2] * __float2half(13)
                                       + data[x][3] * __float2half(14);
            }
        } else {
            if (x > 1 && x < 14) {
                C[offset + x + 224] = data[12][x] * __float2half(2) + data[13][x] * __float2half(7) + data[14][x] * __float2half(-290)
                                    + data[15][x] * __float2half(17) + halo[2][x] * __float2half(22) + data[14][x - 2] * __float2half(10)
                                    + data[14][x - 1] * __float2half(11) + data[14][x + 1] * __float2half(13) + data[14][x + 2] * __float2half(14);
                C[offset + x + 240] = data[13][x] * __float2half(2) + data[14][x] * __float2half(7) + data[15][x] * __float2half(-290)
                                    + halo[2][x] * __float2half(17) + halo[3][x] * __float2half(22) + data[15][x - 2] * __float2half(10)
                                    + data[15][x - 1] * __float2half(11) + data[15][x + 1] * __float2half(13) + data[15][x + 2] * __float2half(14);
                C[offset + x * 16 + 14] = data[x - 2][14] * __float2half(2) + data[x - 1][14] * __float2half(7) + data[x][14] * __float2half(-290)
                                        + data[x + 1][14] * __float2half(17) + data[x + 2][14] * __float2half(22) + data[x][12] * __float2half(10)
                                        + data[x][13] * __float2half(11) + data[x][15] * __float2half(13) + halo[6][x] * __float2half(14);
                C[offset + x * 16 + 15] = data[x - 2][15] * __float2half(2) + data[x - 1][15] * __float2half(7) + data[x][15] * __float2half(-290)
                                        + data[x + 1][15] * __float2half(17) + data[x + 2][15] * __float2half(22) + data[x][13] * __float2half(10)
                                        + data[x][14] * __float2half(11) + halo[6][x] * __float2half(13) + halo[7][x] * __float2half(14);
            } else if (x < 2) {
                tmp[2][0][x] = data[12][x] * __float2half(2) + data[13][x] * __float2half(7) + data[14][x] * __float2half(-290)
                             + data[15][x] * __float2half(17) + halo[2][x] * __float2half(22);
                tmp[2][1][x] = data[13][x] * __float2half(2) + data[14][x] * __float2half(7) + data[15][x] * __float2half(-290)
                             + halo[2][x] * __float2half(17) + halo[3][x] * __float2half(22);
                C[offset + x * 16 + 14] = tmp[1][x][0] + data[x][12] * __float2half(10) + data[x][13] * __float2half(11)
                                        + data[x][15] * __float2half(13) + halo[6][x] * __float2half(14);
                C[offset + x * 16 + 15] = tmp[1][x][1] + data[x][13] * __float2half(10) + data[x][14] * __float2half(11)
                                        + halo[6][x] * __float2half(13) + halo[7][x] * __float2half(14);
            } else {
                tmp[3][0][x - 14] = data[12][x] * __float2half(2) + data[13][x] * __float2half(7) + data[14][x] * __float2half(-290)
                                  + data[15][x] * __float2half(17) + halo[2][x] * __float2half(22);
                tmp[3][1][x - 14] = data[13][x] * __float2half(2) + data[14][x] * __float2half(7) + data[15][x] * __float2half(-290)
                                  + halo[2][x] * __float2half(17) + halo[3][x] * __float2half(22);
                C[offset + x * 16 + 14] = tmp[3][x - 14][0] + data[x][12] * __float2half(10) + data[x][13] * __float2half(11)
                                        + data[x][15] * __float2half(13) + halo[6][x] * __float2half(14);
                C[offset + x * 16 + 15] = tmp[3][x - 14][1] + data[x][13] * __float2half(10) + data[x][14] * __float2half(11)
                                        + halo[6][x] * __float2half(13) + halo[7][x] * __float2half(14);
            }
        }
    }
}