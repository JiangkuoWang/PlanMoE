#include "hetu.h"

HeTuComm::HeTuComm(std::vector<at::cuda::CUDAStream> *stream,
                   std::vector<ncclComm_t>            g_nccl_comm,
                   const int                         &g_world_size,
                   const int                         &g_world_rank,
                   const int                         &g_local_size,
                   const int                         &g_local_rank) :
    AbstractComm(stream, g_nccl_comm, g_world_size, g_world_rank, g_local_size, g_local_rank) {
}

void HeTuComm::pre_comm(const torch::Tensor &input) {
}

void HeTuComm::all_to_all(const torch::Tensor &input, const torch::Tensor &output, size_t length) {

    group_input = torch::empty(
        {input.size(0) * g_local_size, input.size(1)},
        torch::TensorOptions().device(input.device()).dtype(at::kFloat),
        torch::MemoryFormat::Contiguous);
    for (const auto &cuda_stream : *stream) {
        c10::cuda::CUDACachingAllocator::recordStream(group_input.storage().data_ptr(), cuda_stream);
    }
    
    group_output = torch::empty(
        {input.size(0) * g_local_size, input.size(1)},
        torch::TensorOptions().device(input.device()).dtype(at::kFloat),
        torch::MemoryFormat::Contiguous);
    for (const auto &cuda_stream : *stream) {
        c10::cuda::CUDACachingAllocator::recordStream(group_output.storage().data_ptr(), cuda_stream);
    }

    ncclGroupStart();
    if (g_local_rank == 0) {
        for (int i = 0; i < g_local_size; i++) {
            ncclRecv(((char *)group_input.data_ptr()) + i * input.nbytes(), input.nbytes(), ncclInt8, g_world_rank + i, g_nccl_comm[0], stream->at(0).stream());
        }
    }
    ncclSend((char *)input.data_ptr(), input.nbytes(), ncclInt8, g_world_rank - g_local_rank, g_nccl_comm[0], stream->at(0).stream());
    ncclGroupEnd();
    if (g_local_rank == 0) {
        //std::cout << group_input << std::endl;
        _layout_transform(
            group_input,
            group_output,
            g_world_size,
            g_local_size,
            stream->at(0).stream());

        int num_nodes = g_world_size / g_local_size;
        ncclGroupStart();
        for (int i = 0; i < g_world_size; i += g_local_size) {
            ncclRecv(((char *)group_input.data_ptr()) + i * input.nbytes() / num_nodes, group_input.nbytes() / num_nodes, ncclInt8, i, g_nccl_comm[0], stream->at(0).stream());
            ncclSend(((char *)group_output.data_ptr()) + i * input.nbytes() / num_nodes, group_input.nbytes() / num_nodes, ncclInt8, i, g_nccl_comm[0], stream->at(0).stream());
        }
        ncclGroupEnd();

        _reverse_layout_transform(
            group_input,
            group_output,
            g_world_size,
            g_local_size,
            stream->at(0).stream());

        cudaStreamSynchronize(stream->at(0).stream());
    }

    ncclGroupStart();
    if (g_local_rank == 0) {
        for (int i = 0; i < g_local_size; i++) {
            ncclSend(((char *)group_output.data_ptr()) + i * input.nbytes(), input.nbytes(), ncclInt8, g_world_rank + i, g_nccl_comm[0], stream->at(0).stream());
        }
    }
    ncclRecv((char *)output.data_ptr(), input.nbytes(), ncclInt8, g_world_rank - g_local_rank, g_nccl_comm[0], stream->at(0).stream());
    ncclGroupEnd();
}