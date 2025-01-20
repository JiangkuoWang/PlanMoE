#ifndef LZ
#define LZ
#include "abstract.h"
#include "gpulz.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <iostream>
#include <nccl.h>
#include <torch/extension.h>
#include <type_traits>
#include <vector>

class LzCompressor : public AbstractCompressor {
public:
    // Declare all public members here
    torch::Tensor compress(const torch::Tensor &);
    torch::Tensor decompress(const torch::Tensor &);
    void          all_to_all(const torch::Tensor &, const torch::Tensor &);
    LzCompressor(std::shared_ptr<AbstractComm> comm_ptr);

private:
    torch::Tensor flagArrOffsetGlobal, compressedDataOffsetGlobal, flagArrGlobal, compressedDataGlobal;
    torch::Tensor g_flagArrOffsetGlobal, g_compressedDataOffsetGlobal, g_flagArrGlobal, g_compressedDataGlobal;
};
#endif // LZ