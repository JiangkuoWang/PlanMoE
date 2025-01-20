#ifndef HETU_COMM
#define HETU_COMM

#include "abstract.h"
#include "layout_transform.h"
#include <c10/cuda/CUDACachingAllocator.h>

class HeTuComm : public AbstractComm {
public:
    // Declare all public members here
    void all_to_all(const torch::Tensor &, const torch::Tensor &, size_t);
    HeTuComm(std::vector<at::cuda::CUDAStream> *, std::vector<ncclComm_t>, const int &, const int &, const int &, const int &);
    void pre_comm(const torch::Tensor &);
    ~HeTuComm() override = default;

    torch::Tensor group_input, group_output;
};

#endif // HETU_COMM