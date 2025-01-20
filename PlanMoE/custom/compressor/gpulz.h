#ifndef GPU_LZ
#define GPU_LZ
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> lz_compress(torch::Tensor input, cudaStream_t stream);
void                       lz_decompress(
                          torch::Tensor output,
                          int           numOfBlocks,
                          torch::Tensor flagArrOffsetGlobal,
                          torch::Tensor compressedDataOffsetGlobal,
                          torch::Tensor flagArrGlobal,
                          torch::Tensor compressedDataGlobal,
                          cudaStream_t  stream);
#endif // GPU_LZ