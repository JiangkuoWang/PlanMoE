#include "pipe.h"

PipeComm::PipeComm(std::vector<at::cuda::CUDAStream> *stream,
                   std::vector<ncclComm_t>            g_nccl_comm,
                   const int                         &g_world_size,
                   const int                         &g_world_rank,
                   const int                         &g_local_size,
                   const int                         &g_local_rank) :
    AbstractComm(stream, g_nccl_comm, g_world_size, g_world_rank, g_local_size, g_local_rank) {
}

void PipeComm::all_to_all(const torch::Tensor &input, const torch::Tensor &output, size_t length) {
    length = length / g_world_size;
    CHECK_EQ(0, ncclGroupStart());
    for (int i = 0; i < g_world_size; ++i) {
        bool is_intra = (g_world_rank / g_local_size) == (i / g_local_size);
        CHECK_EQ(0, ncclSend(((char *)input.data_ptr()) + i * length,
                             length,
                             ncclInt8,
                             i,
                             g_nccl_comm[is_intra],
                             stream->at(is_intra).stream()));
        CHECK_EQ(0, ncclRecv(((char *)output.data_ptr()) + i * length,
                             length,
                             ncclInt8,
                             i,
                             g_nccl_comm[is_intra],
                             stream->at(is_intra).stream()));
    }
    CHECK_EQ(0, ncclGroupEnd());
}
