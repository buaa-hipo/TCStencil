#include <mma.h>
#include "common/common.hpp"


using namespace nvcuda;

extern "C" void host_code(float *h_A, float *h_B, float *h_C, int N, int n);
extern "C" void gold(float *h_A, float *h_B, float *h_C, int N, int n);

void init_B(float *B){
    for(int i = 0; i < 9; i++){
        B[i] = i;
        // B[i] = 1;
    }
    B[4] = -40;
}


int main() {
    srand(1);
    float *h_A = getRandomArray<float>(MESH_SIZE * MESH_SIZE);
    float *h_B = new float[3 * 3];
    init_B(h_B);
    float *h_C = new float[MESH_SIZE * MESH_SIZE];
    float *C = new float[MESH_SIZE * MESH_SIZE];
    memcpy(C, h_A, sizeof(float) * MESH_SIZE * MESH_SIZE);

    host_code(h_A, h_B, h_C, MESH_SIZE, 3);
    // gold(h_A, h_B, C, MESH_SIZE, 3);
    check_error("all finished");

    // double error = checkError2D(MESH_SIZE, h_C, C, 1, MESH_SIZE - 1, 1, MESH_SIZE - 1);

    // double relative_error = checkRelativeError2D(MESH_SIZE, h_C, C, 1, MESH_SIZE - 1, 1, MESH_SIZE - 1);
    
    // printf("[Test] RMS Error: %e\n", error);
    // printf("[Test] Relative RMS Error: %e\n", relative_error);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] C;
}
