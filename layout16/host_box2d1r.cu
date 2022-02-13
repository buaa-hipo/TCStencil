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


// coe
// * - - - - - - - *
// | * * * * * . . |
// | * * * * * . . |
// | * * * * * . . |
// | * * * * * . . |
// | * * * * * . . |
// | . * * * * * . |
// | . * * * * * . |
// | . * * * * * . |
// | . * * * * * . |
// | . * * * * * . |
// | . . . . . . . |
// * - - - - - - - *
void init_coe(half *__restrict__ B, half *__restrict__ coe, int n) {
    for (int i = 0; i < 16 * 16 * 3; i++) {
        coe[i] = __float2half(0);
    }
    for(int z = 0; z < 3; z++){
        int offset = z * 256;
        for(int i = 0; i < 14; i++){
            for(int j = 0; j < 3; j++){
                coe[offset + 17 * i + 16 * j + 1] = B[z * 3 + j];
            }
        }
        // full part coe
        for (int i = 0; i < 2; i++) {
            coe[offset + i * 16] = B[z * 3 + i + 1];
            coe[offset + i * 16 + 239] = B[z * 3 + i];
        }
    }
}

// * * * * *
// * * * * *
// * * * * *
// * * * * *
// * * * * *
extern "C" __global__ void mma_run(half *__restrict__ A, half *__restrict__ coe,
                               half *__restrict__ C, int N, int tile_size, int *index1, int *index2);


void padding_2D(__half * in, __half * out, int size, int halo_size, __half value) {
    int padding_size = size + halo_size * 2;
    for (int i = 0; i < padding_size; i++) {
        for (int j = 0; j < padding_size; j++) {
            if (halo_size <= i && i < size + halo_size && halo_size <= j && j < size + halo_size) {
                out[i * padding_size + j] = in[(i - halo_size) * size + j - halo_size];
            } else {
                out[i * padding_size + j] = value;
            }
            
        }
    }
}

void brick_layout_2D(__half * in, __half * out, int size, int block_size) {
    int n = size / block_size;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int x = 0; x < block_size; x++) {
                for (int y = 0; y < block_size; y++) {
                    out[(i * n + j) * block_size * block_size + x * block_size + y] = in[(i * block_size + x) * size + j * block_size + y];
                }
            }
        }
    }
}

void reverse_layout_2D(__half * in, __half * out, int size, int block_size) {
    int n = size / block_size;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int x = 0; x < block_size; x++) {
                for (int y = 0; y < block_size; y++) {
                    out[(i * block_size + x) * size + j * block_size + y] = in[(i * n + j) * block_size * block_size + x * block_size + y];
                }
            }
        }
    }
}

void set_halo_index(int *index1, int *index2, int N) {
    index1[0] = 0;
    index1[1] = 17;
    index1[2] = 18;
    index1[3] = 35;

    index2[0] = -N * 16 - 1;
    index2[1] = N * 16 - 256 + 15;
    index2[2] = -N * 16 + 240;
    index2[3] = N * 16 + 256;
}

// data size is N * N. Because of halo, we only update (N-4) * (N-4)
extern "C" void host_code(float *h_A, float *h_B, float *h_C, int N, int n) {
    half h_coe[16 * 16 * 3];
    half *half_B = array_float2half(h_B, 9);
    init_coe(half_B, h_coe, n);

    half *A, *coe;
    half *in = array_float2half(h_A, N * N);
    half *half_A = new half[(N + 32) * (N + 32)];
    padding_2D(in, half_A, N, 16, __float2half(1.0));
    delete[] in;
    half *half_C = new half[(N + 32) * (N + 32)];

    brick_layout_2D(half_A, half_C, N + 32, 16);

    cudaMalloc(&A, sizeof(half) * (N + 32) * (N + 32));
    check_error("Failed to allocate device memory for A");
    cudaMemcpy(A, half_C, sizeof(half) * (N + 32) * (N + 32), cudaMemcpyHostToDevice);

    cudaMalloc(&coe, sizeof(half) * 16 * 16 * 3);
    check_error("Failed to allocate device memory for coe");
    cudaMemcpy(coe, h_coe, sizeof(half) * 16 * 16 * 3, cudaMemcpyHostToDevice);

    half *C;
    cudaMalloc(&C, sizeof(half) * (N + 32) * (N + 32));
    check_error("Failed to allocate device memory for C");
    cudaMemcpy(C, half_A, sizeof(half) * (N + 32) * (N + 32), cudaMemcpyHostToDevice);

    // halo index set
    int h_index1[4], h_index2[4];
    set_halo_index(h_index1, h_index2, N + 32);
    int *index1, *index2;
    cudaMalloc(&index1, sizeof(int) * 4);
    check_error("Failed to allocate device memory for index1");
    cudaMalloc(&index2, sizeof(int) * 4);
    check_error("Failed to allocate device memory for index2");
    cudaMemcpy(index1, h_index1, sizeof(int) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(index2, h_index2, sizeof(int) * 4, cudaMemcpyHostToDevice);

    dim3 blockconfig(16, 2);
    int tile_size = TILE_SIZE;
    dim3 gridconfig(ceil(N, 16 * tile_size), ceil(N, 16));

    cudaEvent_t start, stop;
    float elapsed = 0.0;
    double sum = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int test_t = 0; test_t < RUN_TIMES; test_t++) {
        // cudaMemcpy(A, half_A, sizeof(half) * N * N, cudaMemcpyHostToDevice);
        half *in = A;
        half *out = C;
        cudaDeviceSynchronize();
        cudaEventRecord(start, 0);

        mma_run<<<gridconfig, blockconfig>>>(in, coe, out, N + 32, tile_size, index1, index2);
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        if (test_t >= SKIP_TIMES)
            sum += elapsed;
    }
    printf("[Time] Time used: %f ms\n", sum / (RUN_TIMES - SKIP_TIMES) / STEP_TIMES);
    check_error("finished");

    cudaMemcpy(half_A, C, sizeof(half) * (N + 32) * (N + 32), cudaMemcpyDeviceToHost);

    reverse_layout_2D(half_A, half_C, N + 32, 16);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_C[i * N + j] = __half2float(half_C[(i + 16) * (N + 32) + j + 16]);
        }
    }

    delete[] half_A;
    delete[] half_B;
    delete[] half_C;
    cudaFree(A);
    cudaFree(coe);
    cudaFree(C);
}
