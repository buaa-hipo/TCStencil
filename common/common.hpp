#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mma.h>
#include <sys/time.h>
#include "./param.hpp"


inline void check_error(const char *msg) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error : %s, %s\n", msg, cudaGetErrorString(error));
        exit(-1);
    }
}


template<typename T>
// T get_random() { return ((T)rand()) / (T)((RAND_MAX - 1) / 1) - 0.5; }
T get_random() { return 1; }

template<typename T>
void random_array(T *array, int length) {
    for (int i = 0; i < length; i++) {
        array[i] = get_random<T>();
    }
}

template<typename T>
T *getRandomArray(int length) {
    T *array = new T[length];
    random_array(array, length);
    return array;
}


inline double checkError2D(int N, float *l_output, float *l_reference, int y_lb,
                    int y_ub, int x_lb, int x_ub) {
    float(*output)[N] = (float(*)[N])(l_output);
    float(*reference)[N] = (float(*)[N])(l_reference);
    double error = 0.0;
    double max_error = TOLERANCE;
    int max_k = 0, max_j = 0;
    for (int j = y_lb; j < y_ub; j++)
        for (int k = x_lb; k < x_ub; k++) {
            // printf ("Values at index (%d,%d) are %.6f and %.6f\n", j, k,
            // __T2float(reference[j][k]), __T2float(output[j][k]));
            double curr_error = output[j][k] - reference[j][k];
            curr_error = (curr_error < 0.0 ? -curr_error : curr_error);
            error += curr_error * curr_error;
            if (curr_error > max_error) {
                printf("Values at index (%d,%d) differ : %.6f and %.6f\n", j, k,
                       reference[j][k], output[j][k]);
                max_error = curr_error;
                max_k = k;
                max_j = j;
            }
        }
    if (max_k != 0 || max_j != 0)
        printf("[Test] Max Error : %e @ (,%d,%d)\n", max_error, max_j, max_k);
    error = sqrt(error / ((y_ub - y_lb) * (x_ub - x_lb)));
    return error;
}

inline  double checkRelativeError2D(int N, float *l_output, float *l_reference, int y_lb,
                    int y_ub, int x_lb, int x_ub) {
    float(*output)[N] = (float(*)[N])(l_output);
    float(*reference)[N] = (float(*)[N])(l_reference);
    double error = 0.0;
    double max_error = TOLERANCE;
    int max_k = 0, max_j = 0;
    for (int j = y_lb; j < y_ub; j++)
        for (int k = x_lb; k < x_ub; k++) {
            // printf ("Values at index (%d,%d) are %.6f and %.6f\n", j, k,
            // __T2float(reference[j][k]), __T2float(output[j][k]));
            double curr_error = output[j][k] - reference[j][k];
            curr_error = (abs(reference[j][k]) < MIN_NUMBER ? curr_error / MIN_NUMBER : curr_error / reference[j][k]);
            error += curr_error * curr_error;
            if (curr_error > max_error) {
                printf("Values at index (%d,%d) differ : %.6f and %.6f\n", j, k,
                       reference[j][k], output[j][k]);
                max_error = curr_error;
                max_k = k;
                max_j = j;
            }
        }
    if (max_k != 0 || max_j != 0)
        printf("[Test] Max Relative Error : %e @ (,%d,%d)\n", max_error, max_j, max_k);
    error = sqrt(error / ((y_ub - y_lb) * (x_ub - x_lb)));
    return error;
}

inline  double checkRelativeError3D(int N, float *l_output, float *l_reference, int z_lb,
                    int z_ub, int y_lb, int y_ub, int x_lb, int x_ub) {
    float(*output)[N][N] = (float(*)[N][N])(l_output);
    float(*reference)[N][N] = (float(*)[N][N])(l_reference);
    double error = 0.0;
    double max_error = TOLERANCE;
    int max_x = 0, max_y = 0, max_z = 0;
    for (int z = z_lb; z < z_ub; z++) {
        for (int y = y_lb; y < y_ub; y++) {
            for (int x = x_lb; x < x_ub; x++) {
                double curr_error = output[z][y][x] - reference[z][y][x];
                curr_error = (abs(reference[z][y][x]) < MIN_NUMBER ? curr_error / MIN_NUMBER : curr_error / reference[z][y][x]);
                // printf("%f %f\n", reference[z][y][x], output[z][y][x]);
                // break;
                error += curr_error * curr_error;
                if (curr_error > max_error) {
                    printf("Values at index (%d,%d,%d) differ : %.6f and %.6f\n", z, y, x,
                        reference[z][y][x], output[z][y][x]);
                    max_error = curr_error;
                    max_z = z;
                    max_y = y;
                    max_x = x;
                }
            }
        }
    }
    if (max_x != 0 || max_y != 0 || max_z != 0)
        printf("[Test] Max Relative Error : %e @ (%d,%d,%d)\n", max_error, max_z, max_y, max_x);
    error = sqrt(error / ((z_ub - z_lb) * (y_ub - y_lb) * (x_ub - x_lb)));
    return error;
}

 inline void printf2D(half *h_data, int x, int y, int N, int halo_size) {
    half(*data)[N] = (half(*)[N])(h_data);
    for (int i = x - halo_size; i < x + halo_size + 1; i++) {
        for (int j = y - halo_size; j < y + halo_size + 1; j++) {
            printf("%.2f, ", __half2float(data[i][j]));
        }
        printf("\n");
    }
}

inline void printf2D(half *h_data, int y_low, int y_high, int x_low, int x_high, int N){
    half(*data)[N] = (half(*)[N])(h_data);
    for (int i = y_low; i < y_high; i++) {
        for (int j = x_low; j < x_high; j++) {
            printf("%.2f, ", __half2float(data[i][j]));
        }
        printf("\n");
    }
}

inline half* array_float2half(float *input, int N){
    half* output = new half[N];
    for(int i = 0; i < N; i++){
        output[i] = __float2half(input[i]);
    }
    return output;
}

inline float* array_half2float(half *input, int N){
    float* output = new float[N];
    for(int i = 0; i < N; i++){
        output[i] = __half2float(input[i]);
    }
    return output;
}
