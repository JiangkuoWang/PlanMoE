#ifndef INT8_COMPRESSOR
#define INT8_COMPRESSOR

#include "abstract.h"
#include <iostream>
#include <nccl.h>
#include <torch/extension.h>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

class Int8Compressor : public AbstractCompressor {
public:
    // Declare all public members here
    torch::Tensor compress(const torch::Tensor &);
    torch::Tensor decompress(const torch::Tensor &);
    void          all_to_all(const torch::Tensor &, const torch::Tensor &);
    void          pre_comm(const at::cuda::CUDAStream *);
    Int8Compressor(std::shared_ptr<AbstractComm>);
    ~Int8Compressor() = default;

private:
    torch::Tensor     bias, scale, g_bias, g_scale;
    caffe2::TypeMeta  dtype;
    size_t            length;
    std::vector<long> sizes;
};

#endif // INT8_COMPRESSOR