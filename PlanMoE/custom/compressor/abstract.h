#ifndef ABSTRACT_COMPRESSOR
#define ABSTRACT_COMPRESSOR

#include "../comm/abstract.h"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <nccl.h>
#include <torch/extension.h>
#include <type_traits>

class AbstractCompressor {
public:
    virtual torch::Tensor compress(const torch::Tensor &)                          = 0;
    virtual torch::Tensor decompress(const torch::Tensor &)                        = 0;
    virtual void          all_to_all(const torch::Tensor &, const torch::Tensor &) = 0;
    virtual void          pre_comm(const at::cuda::CUDAStream *);
    AbstractCompressor(std::shared_ptr<AbstractComm>);
    virtual ~AbstractCompressor();

    torch::Tensor                 g_output;
    std::shared_ptr<AbstractComm> comm_ptr;
};

#endif // ABSTRACT_COMPRESSOR