#include "device_launch_parameters.h"
#include "stdio.h"
#include <cuda.h>
#include <iostream>
#include <mma.h>
#include <vector>
#include <cooperative_groups.h>
#include "../common/common.hpp"

#define ceil(a, b) ((a) % (b) == 0 ? (a) / (b) : ((a) / (b)) + 1)

using namespace nvcuda;


// * * * * *
// * * * * *
// * * * * *
// * * * * *
// * * * * *
//处理25点stencil，半径r=2
extern "C" __global__ void mma_run(half *__restrict__ A, half *__restrict__ coe, half *__restrict__ C, int N, int tile_size, int *index1, int *index2) {
    __shared__ half data[320];
    __shared__ half halo[18 * 2];

    const int index = (threadIdx.y << 4) + threadIdx.x;
    const int offset_base = ((blockIdx.y + 1) << 4) * N + ((blockIdx.x * tile_size + 1) << 8);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_list[3];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    #pragma unroll (3)
    for(int i = 0; i < 3; i++) {
        wmma::load_matrix_sync(b_frag_list[i], coe + (i << 8), 16);
    }

    #pragma unroll
    for(int iter = 0; iter < tile_size; iter++) {
        int offset = offset_base + (iter << 8);
        // load data
        ((float4 *)(data + 32))[index] =  ((float4 *)(A + offset))[index];
        // #pragma unroll
        // for (int i = 0; i < 8; i++) {
        //     data[(i << 5) + index + 32] = A[(i << 5) + offset + index];
        // }
        data[index] = A[offset + index - (N << 4) + 240];
        data[index + 288] = A[(N << 4) + offset + index];
        // load halo
        halo[threadIdx.y * 18 + threadIdx.x + 1] =  A[(threadIdx.x << 4) + offset - 241 + threadIdx.y * 497];
        if (index < 4) {
            halo[index1[index]] = A[offset + index2[index]];
        }

        wmma::fill_fragment(c_frag, 0.0f);

        #pragma unroll (3)
        for(int i = 0; i < 3; i++) {
            wmma::load_matrix_sync(a_frag, data + (i << 4), 16);
            wmma::mma_sync(c_frag, a_frag, b_frag_list[i], c_frag);
        }
        wmma::store_matrix_sync(data, c_frag, 16, wmma::mem_row_major);
        // left
        if (threadIdx.y == 0) {
            data[threadIdx.x << 4] += halo[threadIdx.x] * __float2half(0) + halo[threadIdx.x + 1] * __float2half(3) + halo[threadIdx.x + 2] * __float2half(6);
        } // right
        else {
            data[(threadIdx.x << 4) + 15] += halo[threadIdx.x + 16] * __float2half(2) + halo[threadIdx.x + 17] * __float2half(5) + halo[threadIdx.x + 18] * __float2half(8);
        }
        __syncthreads();
        ((float4 *)(C + offset))[index] = ((float4 *)data)[index];
    }
}
