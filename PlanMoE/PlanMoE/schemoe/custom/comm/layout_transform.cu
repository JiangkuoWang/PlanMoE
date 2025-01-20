#include "layout_transform.h"

__global__ void layout_transform_kernel(const float *input_data, float *output_data, int samples, int hidden, int g_world_size, int g_local_size) {
    int num_nodes                  = g_world_size / g_local_size;
    int data_size_per_gpu          = samples / g_local_size;
    int data_size_per_gpu_per_node = data_size_per_gpu / (num_nodes);
    int data_size_per_gpu_per_gpu  = data_size_per_gpu / (g_world_size);
    int data_size_per_node         = samples / num_nodes;
    int gpu_id                     = 0;
    int target_node_id             = 0;
    int target_gpu_id              = 0;
    int tmp                        = 0;
    int offset                     = 0;
    for (int i = blockIdx.x; i < samples; i += gridDim.x) {
        gpu_id         = i / data_size_per_gpu;
        tmp            = i % data_size_per_gpu;
        target_node_id = tmp / data_size_per_gpu_per_node;
        tmp            = tmp % data_size_per_gpu_per_node;
        target_gpu_id  = tmp / data_size_per_gpu_per_gpu;
        offset         = tmp % data_size_per_gpu_per_gpu;
        for (int j = threadIdx.x; j < hidden; j += 1024) {
            output_data[(target_node_id * data_size_per_node + target_gpu_id * data_size_per_gpu_per_node + gpu_id * data_size_per_gpu_per_gpu + offset) * (hidden) + j] = input_data[i * (hidden) + j];
        }
    }
}

__global__ void reverse_layout_transform_kernel(const float *input_data, float *output_data, int samples, int hidden, int g_world_size, int g_local_size) {
    int num_nodes                  = g_world_size / g_local_size;
    int data_size_per_gpu          = samples / g_local_size;
    int data_size_per_gpu_per_node = data_size_per_gpu / (num_nodes);
    int data_size_per_gpu_per_gpu  = data_size_per_gpu / (g_world_size);
    int data_size_per_node         = samples / num_nodes;
    int gpu_id                     = 0;
    int target_node_id             = 0;
    int target_gpu_id              = 0;
    int tmp                        = 0;
    int offset                     = 0;
    for (int i = blockIdx.x; i < samples; i += gridDim.x) {
        target_node_id = i / data_size_per_node;
        tmp            = i % data_size_per_node;
        target_gpu_id  = tmp / data_size_per_gpu_per_node;
        tmp            = tmp % data_size_per_gpu_per_node;
        gpu_id         = tmp / data_size_per_gpu_per_gpu;
        offset         = tmp % data_size_per_gpu_per_gpu;
        for (int j = threadIdx.x; j < hidden; j += 1024) {
            output_data[(target_gpu_id * data_size_per_gpu + target_node_id * data_size_per_gpu_per_node + gpu_id * data_size_per_gpu_per_gpu + offset) * (hidden) + j] = input_data[i * (hidden) + j];
        }
    }
}

void _layout_transform(
    torch::Tensor input,
    torch::Tensor output,
    int           g_world_size,
    int           g_local_size,
    cudaStream_t  stream) {
    int          samples     = input.size(0);
    int          hidden      = input.size(1);
    const float *input_data  = (const float *)(input.data_ptr());
    float       *output_data = (float *)(output.data_ptr());
    dim3         blocks;
    dim3         threads;
    blocks.x  = 128;
    threads.x = 1024;
    layout_transform_kernel<<<blocks, threads, 0, stream>>>(input_data, output_data, samples, hidden, g_world_size, g_local_size);
}

void _reverse_layout_transform(
    torch::Tensor input,
    torch::Tensor output,
    int           g_world_size,
    int           g_local_size,
    cudaStream_t  stream) {
    int          samples     = input.size(0);
    int          hidden      = input.size(1);
    const float *input_data  = (const float *)(input.data_ptr());
    float       *output_data = (float *)(output.data_ptr());
    dim3         blocks;
    dim3         threads;
    blocks.x  = 128;
    threads.x = 1024;
    reverse_layout_transform_kernel<<<blocks, threads, 0, stream>>>(input_data, output_data, samples, hidden, g_world_size, g_local_size);
}
