#include <stdio.h>

extern "C" void gold(float *h_A, float *h_B, float *h_C, int N, int n) {
    float(*A)[N] = (float(*)[N])h_A;
    float(*B)[5] = (float(*)[5])h_B;
    float(*C)[N] = (float(*)[N])h_C;

    for (int i = 2; i < N - 2; i++) {
        for (int j = 2; j < N - 2; j++) {
            C[i][j] = 0;
            for(int k = 0; k < 25; k++){
                C[i][j] += A[i - 2 + k / 5][j - 2 + k % 5] * B[k / 5][k % 5];
            }
        }
    }
    // printf("C\n");
    // for(int i = 0; i < 5; i++){
    //     for(int j = 0; j < 5; j++){
    //         printf("%.6f ", C[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
}