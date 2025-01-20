#include "abstract.h"

AbstractComm::AbstractComm(std::vector<at::cuda::CUDAStream> *stream,
                           std::vector<ncclComm_t>            g_nccl_comm,
                           const int                         &g_world_size,
                           const int                         &g_world_rank,
                           const int                         &g_local_size,
                           const int                         &g_local_rank) :
    stream(stream),
    g_nccl_comm(g_nccl_comm),
    g_world_size(g_world_size),
    g_world_rank(g_world_rank),
    g_local_size(g_local_size),
    g_local_rank(g_local_rank) {
}

void AbstractComm::pre_comm(const torch::Tensor &input) {
}

AbstractComm::~AbstractComm() {
}