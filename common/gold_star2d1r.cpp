#include <stdio.h>
#include "param.hpp"

extern "C" void gold(float *h_A, float *h_B, float *h_C, int N, int n) {
    float(*A)[N] = (float(*)[N])h_A;
    float(*B)[3] = (float(*)[3])h_B;
    float(*C)[N] = (float(*)[N])h_C;
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            C[i][j] = A[i - 1][j] * B[0][1] + A[i][j] * B[1][1] +
                    A[i + 1][j] * B[2][1] + A[i][j - 1] * B[1][0] +
                    A[i][j + 1] * B[1][2];
        }
    }
    // for(int i = 0; i < 5; i++){
    //     for(int j = 0; j < 5; j++){
    //         printf("%.2f ", C[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
}