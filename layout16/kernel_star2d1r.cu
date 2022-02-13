#include "device_launch_parameters.h"
#include "stdio.h"
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <vector>
#include "../common/common.hpp"

#define ceil(a, b) ((a) % (b) == 0 ? (a) / (b) : ((a) / (b)) + 1)

using namespace nvcuda;

//     *
//     *
// * * * * *
//     *
//     *
extern "C" __global__ void mma_run(half *__restrict__ A, half *__restrict__ coe_a, half *__restrict__ coe_b, half *__restrict__ C, int N, int tile_size, int *index1, int *index2) {
    __shared__ half data[256];
    __shared__ half halo[64];
    const int index = threadIdx.x + (threadIdx.y << 4);
    const int offset_base = (blockIdx.y + 1) * (N << 4) + ((blockIdx.x * tile_size + 1) << 8);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::load_matrix_sync(a_frag[0], coe_a, 16);
    wmma::load_matrix_sync(b_frag[1], coe_b, 16);

    #pragma unroll
    for(int i = 0; i < tile_size; i++){
        int offset = offset_base + (i << 8);

        halo[index] = A[offset + index2[index]];
        halo[index + 32] = A[offset + index2[index + 32]];

        wmma::load_matrix_sync(b_frag[0], A + offset, 16);
        wmma::load_matrix_sync(a_frag[1], A + offset, 16);
        wmma::fill_fragment(c_frag, 0.0f);
        wmma::mma_sync(c_frag, a_frag[0], b_frag[0], c_frag);
        wmma::mma_sync(c_frag, a_frag[1], b_frag[1], c_frag);
        wmma::store_matrix_sync(data, c_frag, 16, wmma::mem_row_major);
        // do halo compute
        // top and left
        if (threadIdx.y == 0) {
            data[threadIdx.x] += halo[threadIdx.x] * __float2half(1);

            data[index2[index + 32] + 241] += halo[threadIdx.x + 32] * __float2half(3);
        } // bottom and right
        else {
            data[threadIdx.x + 240] += halo[threadIdx.x + 16] * __float2half(7);

            data[index2[index + 16] + 256] += halo[threadIdx.x + 48] * __float2half(5);
        }
        __syncthreads();
        // write
        ((float4 *)(C + offset))[index] = ((float4 *)data)[index];
        // #pragma unroll
        // for (int j = 0; j < 8; j++){
        //     C[index + offset + (j << 5)] = data[index + (j << 5)];
        // }
    }
}
