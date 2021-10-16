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
    __shared__ half halo[80];

    const int index = (threadIdx.y << 4) + threadIdx.x;
    const int offset_base = ((blockIdx.y + 1) << 4) * N + ((blockIdx.x * tile_size + 1) << 8);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag_list[5];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    #pragma unroll (5)
    for(int i = 0; i < 5; i++) {
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
        data[index] = A[offset + index - (N << 4) + 224];
        data[index + 288] = A[(N << 4) + offset + index];
        // load halo
        halo[threadIdx.y * 20 + threadIdx.x + 2] =  A[(threadIdx.x << 4) + offset - 242 + threadIdx.y];
        halo[threadIdx.y * 20 + threadIdx.x + 42] = A[(threadIdx.x << 4) + offset + 256 + threadIdx.y];
        if (threadIdx.y == 0) {
            halo[index1[index]] = A[offset + index2[index]];
        }

        wmma::fill_fragment(c_frag, 0.0f);

        #pragma unroll (5)
        for(int i = 0; i < 5; i++) {
            wmma::load_matrix_sync(a_frag, data + (i << 4), 16);
            wmma::mma_sync(c_frag, a_frag, b_frag_list[i], c_frag);
        }
        wmma::store_matrix_sync(data, c_frag, 16, wmma::mem_row_major);
        // left
        if (threadIdx.y == 0) {
            // data[(threadIdx.x << 4)] += halo[threadIdx.x] * __float2half(0) + halo[threadIdx.x + 1] * __float2half(5) + halo[threadIdx.x + 2] * __float2half(10)
            //                       + halo[threadIdx.x + 3] * __float2half(15) + halo[threadIdx.x + 4] * __float2half(20)
            //                       + halo[threadIdx.x + 20] * __float2half(1) + halo[threadIdx.x + 21] * __float2half(6) + halo[threadIdx.x + 22] * __float2half(11)
            //                       + halo[threadIdx.x + 23] * __float2half(16) + halo[threadIdx.x + 24] * __float2half(21);
            // data[(threadIdx.x << 4) + 1] += halo[threadIdx.x + 20] * __float2half(0) + halo[threadIdx.x + 21] * __float2half(5) + halo[threadIdx.x + 22] * __float2half(10)
            //                               + halo[threadIdx.x + 23] * __float2half(15) + halo[threadIdx.x + 24] * __float2half(20);

            const int tmp_index = (threadIdx.x << 4);
            data[tmp_index] += halo[threadIdx.x] * __float2half(0) + halo[threadIdx.x + 1] * __float2half(5) + halo[threadIdx.x + 2] * __float2half(10)
                             + halo[threadIdx.x + 3] * __float2half(15) + halo[threadIdx.x + 4] * __float2half(20);

            data[tmp_index]     += halo[threadIdx.x + 20] * __float2half(1);
            data[tmp_index + 1] += halo[threadIdx.x + 20] * __float2half(0);
            data[tmp_index]     += halo[threadIdx.x + 21] * __float2half(6);
            data[tmp_index + 1] += halo[threadIdx.x + 21] * __float2half(5);
            data[tmp_index]     += halo[threadIdx.x + 22] * __float2half(11);
            data[tmp_index + 1] += halo[threadIdx.x + 22] * __float2half(10);
            data[tmp_index]     += halo[threadIdx.x + 23] * __float2half(16);
            data[tmp_index + 1] += halo[threadIdx.x + 23] * __float2half(15);
            data[tmp_index]     += halo[threadIdx.x + 24] * __float2half(21);
            data[tmp_index + 1] += halo[threadIdx.x + 24] * __float2half(20);

            // const int tmp_index = (threadIdx.x << 3);
            // data[tmp_index << 1] += halo[threadIdx.x] * __float2half(0) + halo[threadIdx.x + 1] * __float2half(5) + halo[threadIdx.x + 2] * __float2half(10)
            //                       + halo[threadIdx.x + 3] * __float2half(15) + halo[threadIdx.x + 4] * __float2half(20);

            // ((__half2 *)(data))[tmp_index] += __halves2half2(__float2half(1), __float2half(0));
            // ((__half2 *)(data))[tmp_index] += __halves2half2(__float2half(6), __float2half(5));
            // ((__half2 *)(data))[tmp_index] += __halves2half2(__float2half(11), __float2half(10));
            // ((__half2 *)(data))[tmp_index] += __halves2half2(__float2half(16), __float2half(15));
            // ((__half2 *)(data))[tmp_index] += __halves2half2(__float2half(21), __float2half(20));
        } // right
        else {
            // data[(threadIdx.x << 4) + 15] += halo[threadIdx.x + 40] * __float2half(3) + halo[threadIdx.x + 41] * __float2half(8) + halo[threadIdx.x + 42] * __float2half(13)
            //                            + halo[threadIdx.x + 43] * __float2half(18) + halo[threadIdx.x + 44] * __float2half(23)
            //                            + halo[threadIdx.x + 60] * __float2half(4) + halo[threadIdx.x + 61] * __float2half(9) + halo[threadIdx.x + 62] * __float2half(14)
            //                            + halo[threadIdx.x + 63] * __float2half(19) + halo[threadIdx.x + 64] * __float2half(24);
            // data[(threadIdx.x << 4) + 14] += halo[threadIdx.x + 40] * __float2half(4) + halo[threadIdx.x + 41] * __float2half(9) + halo[threadIdx.x + 42] * __float2half(14)
            //                             + halo[threadIdx.x + 43] * __float2half(19) + halo[threadIdx.x + 44] * __float2half(24);
            
            const int tmp_index = (threadIdx.x << 4) + 14;
            data[tmp_index]     += halo[threadIdx.x + 40] * __float2half(4);
            data[tmp_index + 1] += halo[threadIdx.x + 40] * __float2half(3);
            data[tmp_index]     += halo[threadIdx.x + 41] * __float2half(9);
            data[tmp_index + 1] += halo[threadIdx.x + 41] * __float2half(8);
            data[tmp_index]     += halo[threadIdx.x + 42] * __float2half(14);
            data[tmp_index + 1] += halo[threadIdx.x + 42] * __float2half(13);
            data[tmp_index]     += halo[threadIdx.x + 43] * __float2half(19);
            data[tmp_index + 1] += halo[threadIdx.x + 43] * __float2half(18);
            data[tmp_index]     += halo[threadIdx.x + 44] * __float2half(24);
            data[tmp_index + 1] += halo[threadIdx.x + 44] * __float2half(23);

            data[tmp_index + 1] += halo[threadIdx.x + 60] * __float2half(4) + halo[threadIdx.x + 61] * __float2half(9) + halo[threadIdx.x + 62] * __float2half(14)
                                + halo[threadIdx.x + 63] * __float2half(19) + halo[threadIdx.x + 64] * __float2half(24);

            // const int tmp_index = (threadIdx.x << 3) + 7;
            // ((__half2 *)(data))[tmp_index] += __halves2half2(__float2half(4), __float2half(3));
            // ((__half2 *)(data))[tmp_index] += __halves2half2(__float2half(9), __float2half(8));
            // ((__half2 *)(data))[tmp_index] += __halves2half2(__float2half(14), __float2half(13));
            // ((__half2 *)(data))[tmp_index] += __halves2half2(__float2half(19), __float2half(18));
            // ((__half2 *)(data))[tmp_index] += __halves2half2(__float2half(24), __float2half(23));
            // data[(tmp_index << 1) + 1] += halo[threadIdx.x + 60] * __float2half(4) + halo[threadIdx.x + 61] * __float2half(9) + halo[threadIdx.x + 62] * __float2half(14)
            //                     + halo[threadIdx.x + 63] * __float2half(19) + halo[threadIdx.x + 64] * __float2half(24);
        }
        __syncthreads();
        ((float4 *)(C + offset))[index] = ((float4 *)data)[index];
        // #pragma unroll (8)
        // for (int i = 0; i < 8; i++) {
        //     C[(i << 5) + offset + index] = data[(i << 5) + index];
        // }
    }
}
