#include <stdio.h>
#include <math.h>

#define MAX 8192
#define LOG_MAX 13
#define BLOCK_SIZE 512
#define N 10

__host__ void fftHost(float x_r[], float x_i[])
{
    float tmp_r, tmp_i ;
    int i, j, i_lower ;
    int stage, dft_pts, num_bf;
    float pi;

    pi = -2 * M_PI;

    float arg, e, cos_result, sin_result;

    for (stage = 1; stage <= LOG_MAX; stage++) {
        dft_pts = 1 << stage;
        num_bf = dft_pts / 2;
        e = pi / dft_pts;
        for (j = 0; j < num_bf; j++) {
            arg = e * j;
            cos_result = cos(arg);
            sin_result = sin(arg);
            for (i = j; i < MAX; i += dft_pts) {
                i_lower = i + num_bf;

                tmp_r = x_r[i_lower] * cos_result - x_i[i_lower] * sin_result;
                tmp_i = x_i[i_lower] * cos_result + x_r[i_lower] * sin_result;
                x_r[i_lower] = x_r[i] - tmp_r;
                x_i[i_lower] = x_i[i] - tmp_i;
                x_r[i] = x_r[i] + tmp_r;
                x_i[i] = x_i[i] + tmp_i;
            }
        }
    }
}

__global__ void fftKernel(float *dx_r, float *dx_i) {
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    float tmp_r, tmp_i;
    int i, j, i_lower;
    int dft_pts, num_bf;
    float pi;
    float arg, e, cos_result, sin_result;
    pi = -2 * M_PI;
    for (int stage = 1; stage <= 13; stage++) {
        dft_pts = 1 << stage;
        num_bf = dft_pts / 2;
        e = pi / dft_pts;
        if (thread_id < BLOCK_SIZE) {
            int start = MAX/(2*BLOCK_SIZE) * thread_id;
            for (int k = 0; k < MAX/(2*BLOCK_SIZE); k++) {
                i = (start + k) + int(pow(2, stage - 1)) * ((start + k) / int(pow(2, stage - 1)));
                j = i % num_bf;
                arg = e * j;
                cos_result = cos(arg);
                sin_result = sin(arg);
                i_lower = i + num_bf;
                tmp_r = dx_r[i_lower] * cos_result - dx_i[i_lower] * sin_result;
                tmp_i = dx_i[i_lower] * cos_result + dx_r[i_lower] * sin_result;
                dx_r[i_lower] = dx_r[i] - tmp_r;
                dx_i[i_lower] = dx_i[i] - tmp_i;
                dx_r[i] = dx_r[i] + tmp_r;
                dx_i[i] = dx_i[i] + tmp_i;
            }
        }
        __syncthreads();
    }
}

