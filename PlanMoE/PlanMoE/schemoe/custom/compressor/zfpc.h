#ifndef ZFP_COMPRESSOR
#define ZFP_COMPRESSOR

#include "abstract.h"
#include "assert.h"
#include "zfp.h"
#include <c10/cuda/CUDACachingAllocator.h>

class ZfpCompressor : public AbstractCompressor {
public:
    // Declare all public members here
    torch::Tensor compress(const torch::Tensor &);
    torch::Tensor decompress(const torch::Tensor &);
    void          all_to_all(const torch::Tensor &, const torch::Tensor &);
    void          set_cuda_stream(const cudaStream_t &);
    void          set_compress_rate(const double &);
    ZfpCompressor(std::shared_ptr<AbstractComm>, double = 8.0) noexcept;
    ~ZfpCompressor();

public:
    double            compress_rate;
    void             *buffer;
    size_t            last_bufsize;
    std::vector<long> sizes;
    size_t            length;
};

#endif // ZFP_COMPRESSOR