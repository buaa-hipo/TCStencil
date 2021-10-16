#include <stdio.h>
#include "param.hpp"

extern "C" void gold(float *h_A, float *h_B, float *h_C, int N, int n) {
    float(*A)[N] = (float(*)[N])h_A;
    float(*B)[5] = (float(*)[5])h_B;
    float(*C)[N] = (float(*)[N])h_C;
    for (int i = 2; i < N - 2; i++) {
        for (int j = 2; j < N - 2; j++) {
            C[i][j] = A[i - 2][j] * B[0][2] + A[i - 1][j] * B[1][2] +
                    A[i][j] * B[2][2] + A[i + 1][j] * B[3][2] +
                    A[i + 2][j] * B[4][2] + A[i][j - 2] * B[2][0] +
                    A[i][j - 1] * B[2][1] + A[i][j + 1] * B[2][3] +
                    A[i][j + 2] * B[2][4];
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