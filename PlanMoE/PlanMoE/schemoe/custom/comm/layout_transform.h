#ifndef LAYOUT_TRANSFORM
#define LAYOUT_TRANSFORM

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

extern "C" void _layout_transform(
    torch::Tensor input,
    torch::Tensor output,
    int           g_world_size,
    int           g_local_size,
    cudaStream_t  stream);

extern "C" void _reverse_layout_transform(
    torch::Tensor input,
    torch::Tensor output,
    int           g_world_size,
    int           g_local_size,
    cudaStream_t  stream);

#endif // LAYOUT_TRANSFORM