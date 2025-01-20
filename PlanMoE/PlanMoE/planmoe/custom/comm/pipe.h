#ifndef PIPE_COMM
#define PIPE_COMM

#include "abstract.h"

#undef CHECK_EQ

#define CHECK_EQ(x, y) AT_ASSERTM((x) == (y), "CHECK_EQ fails.")


class PipeComm : public AbstractComm {
public:
    // Declare all public members here
    void all_to_all(const torch::Tensor &, const torch::Tensor &, size_t);
    PipeComm(std::vector<at::cuda::CUDAStream> *, std::vector<ncclComm_t>, const int &, const int &, const int &, const int &);
};

#endif // PIPE_COMM
