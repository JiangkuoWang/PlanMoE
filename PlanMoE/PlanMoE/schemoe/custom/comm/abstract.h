#ifndef ABSTRACT_COMM
#define ABSTRACT_COMM

#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <nccl.h>
#include <torch/extension.h>
#include <type_traits>

class AbstractComm {
public:
    // Declare all public members here
    virtual void all_to_all(const torch::Tensor &, const torch::Tensor &, size_t) = 0;
    AbstractComm(std::vector<at::cuda::CUDAStream> *, std::vector<ncclComm_t>, const int &, const int &, const int &, const int &);
    virtual void pre_comm(const torch::Tensor &);
    virtual ~AbstractComm();

    std::vector<at::cuda::CUDAStream> *stream;
    std::vector<ncclComm_t>            g_nccl_comm;
    int                                g_world_size;
    int                                g_world_rank;
    int                                g_local_size;
    int                                g_local_rank;
};

#endif // ABSTRACT_COMM