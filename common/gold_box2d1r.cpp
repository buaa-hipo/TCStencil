#include <stdio.h>

extern "C" void gold(float *h_A, float *h_B, float *h_C, int N, int n) {
    float(*A)[N] = (float(*)[N])h_A;
    float(*B)[3] = (float(*)[3])h_B;
    float(*C)[N] = (float(*)[N])h_C;

    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            C[i][j] = 0;
            for(int k = 0; k < 9; k++){
                C[i][j] += A[i - 1 + k / 3][j - 1 + k % 3] * B[k / 3][k % 3];
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