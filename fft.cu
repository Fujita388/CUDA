#include <stdio.h>
#include <math.h>


#define MAX 8192
#define LOG_MAX 13
#define N 10


const int BLOCK_SIZE = 512;
void bit_reverse(float x_r[], float x_i[]);


__host__ void fftHost(float *x_r, float *x_i);
__global__ void fftKernel(float *dx_r, float *dx_i);


int main() {
    float *x_r, *x_i, *xr, *xi;
    float *dx_r, *dx_i;
    int i;
    int correct_flag = 1;
    dim3 dim_grid(1, 1), dim_block(BLOCK_SIZE, 1, 1);
    cudaEvent_t start, stop;
    float elapsed_time;

//Step 0. Timer
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    x_r = (float *)malloc(sizeof(float) * MAX);
    x_i = (float *)malloc(sizeof(float) * MAX);
    xr = (float *)malloc(sizeof(float) * MAX);
    xi = (float *)malloc(sizeof(float) * MAX);

    //initialization
    for (i = 0; i < MAX; i++) {
        x_r[i] = cos(N * 2 * M_PI * i / MAX);
        x_i[i] = 0;
    }

    bit_reverse(x_r, x_i);
    cudaEventRecord(start, 0);
    fftHost(x_r, x_i);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    for (i = 0; i < MAX; i++) {
        if (i == N || i == MAX - N) {
            if (round(x_r[i]) != MAX/2 || round(x_i[i]) != 0)
                correct_flag = 0;
        } else {
            if (round(x_r[i]) != 0 || round(x_i[i]) != 0)
                correct_flag = 0;
        }
    }

    if (correct_flag == 1) {
        printf("CPU time[sec]:%lf\n", elapsed_time);
    } else {
        fprintf(stderr, "CPU Failed\n");
    }

    //GPU initialization again
    for (i = 0; i < MAX; i++) {
        x_r[i] = cos(N * 2 * M_PI * i / MAX);
        x_i[i] = 0;
    }
    cudaMalloc((void **)&dx_r, sizeof(float) * MAX);
    cudaMalloc((void **)&dx_i, sizeof(float) * MAX);

    bit_reverse(x_r, x_i);
    cudaMemcpy(dx_r, x_r, sizeof(float) * MAX, cudaMemcpyHostToDevice);
    cudaMemcpy(dx_i, x_i, sizeof(float) * MAX, cudaMemcpyHostToDevice);
    cudaMemcpy(xr, dx_r, sizeof(float) * MAX, cudaMemcpyDeviceToHost);
    cudaMemcpy(xi, dx_i, sizeof(float) * MAX, cudaMemcpyDeviceToHost);
    cudaEventRecord(start, 0);

    fftKernel<<<dim_grid, dim_block>>>(dx_r, dx_i);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaMemcpy(x_r, dx_r, sizeof(float) * MAX, cudaMemcpyDeviceToHost);
    cudaMemcpy(x_i, dx_i, sizeof(float) * MAX, cudaMemcpyDeviceToHost);
    cudaFree(dx_r);
    cudaFree(dx_i);

    correct_flag=1;
    for (i = 0; i < MAX; i++) {
        if (i == N || i == MAX - N) {
            if (round(x_r[i]) != MAX/2 || round(x_i[i]) != 0)
                correct_flag = 0;
        } else {
            if (round(x_r[i]) != 0 || round(x_i[i]) != 0)
                correct_flag = 0;
        }
    }
    if (correct_flag == 1) {
        printf("GPU time[sec]:%lf\n", elapsed_time);
    } else {
        printf("GPU Failed:%lf\n", elapsed_time);
        fprintf(stderr, "GPU Failed:%lf\n", elapsed_time);
    }
}


unsigned int reverse_bits(unsigned int input) {
    unsigned int rev = 0;
    int i;

    for (i = 0; i < LOG_MAX; i++) {
        rev = (rev << 1) | (input & 1);
        input = input >> 1;
    }
    return rev;
}


void bit_reverse(float x_r[], float x_i[]) {
    unsigned int reversed, i;
    float tmp;

    for (i = 0; i < MAX; i++) {
        reversed = reverse_bits(i);
        if (i < reversed) {
            tmp = x_r[i];
            x_r[i] = x_r[reversed];
            x_r[reversed] = tmp;

            tmp = x_i[i];
            x_i[i] = x_i[reversed];
            x_i[reversed] = tmp;
        }
    }
}
