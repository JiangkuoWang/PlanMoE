#ifndef NO_COMPRESSOR
#define NO_COMPRESSOR

#include "abstract.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <iostream>
#include <nccl.h>
#include <torch/extension.h>
#include <type_traits>

class NoCompressor : public AbstractCompressor {
public:
    // Declare all public members here
    torch::Tensor compress(const torch::Tensor &);
    torch::Tensor decompress(const torch::Tensor &);
    void          all_to_all(const torch::Tensor &, const torch::Tensor &);
    NoCompressor(std::shared_ptr<AbstractComm> comm_ptr);
};

#endif // NO_COMPRESSOR