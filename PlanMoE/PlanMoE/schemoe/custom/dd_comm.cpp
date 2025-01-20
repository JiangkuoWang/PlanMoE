#include "dd_comm.h"

DdComm::DdComm(std::vector<at::cuda::CUDAStream> *stream,
                     std::vector<ncclComm_t>            g_nccl_comm,
                     const int &g_world_size, const int &g_world_rank,
                     const int &g_local_size, const int &g_local_rank)
    : AbstractComm(stream, g_nccl_comm, g_world_size, g_world_rank,
                   g_local_size, g_local_rank) {}

extern int jit::mem_stride_copy_char_fd;
extern int jit::mem_stride_copy_uint4_fd;
extern int jit::mem_stride_copy_gridsize;
extern int jit::mem_stride_copy_blocksize;

void DdComm::all_to_all(const torch::Tensor &input, const torch::Tensor &output, size_t length) {
    size_t slice_size       = length / g_world_size;
    size_t slice_size_uint4 = slice_size / sizeof(uint4);

    // Save original stream and switch to NCCL stream
    // Output tensors must be allocated in NCCL stream context to prevent PyTorch Caching Allocator from recycling it
    // const at::cuda::CUDAStream &original_stream = at::cuda::getCurrentCUDAStream();
    // at::cuda::setCurrentCUDAStream(get_nccl_stream());

    // // Computation stream allocator will add blocking event to nccl stream after nccl kernels
    // c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), get_nccl_stream());

    int nranks = g_world_size, ngpus = 4;
    CHECK_EQ(0, nranks % ngpus);
    int nnodes = nranks / ngpus;

    // torch::Tensor tmp_output      = torch::empty_like(input, torch::MemoryFormat::Contiguous);
    void *input_buff      = (void *)input.data_ptr();
    void *output_buff = (void *)output.data_ptr();

    if (!(ngpus == 1 || nnodes == 1)) {
        int node_rank = g_world_rank / ngpus, local_rank = g_local_rank;
        // phase 0. per-gpu (ngpus) stride copy
        //std::cout << jit::mem_stride_copy_char_fd << " " << jit::mem_stride_copy_gridsize << " " << jit::mem_stride_copy_blocksize << " " << jit::mem_stride_copy_uint4_fd << std::endl;
        slice_size < sizeof(uint4) ? jit::jit_execute(
            {&output_buff, &input_buff, &slice_size, &ngpus, &nnodes}, jit::mem_stride_copy_char_fd,
            input.device().index(), jit::mem_stride_copy_gridsize, jit::mem_stride_copy_blocksize, stream->at(0).stream()) :
                                     jit::jit_execute(
                                         {&output_buff, &input_buff, &slice_size_uint4, &ngpus, &nnodes}, jit::mem_stride_copy_uint4_fd,
                                         input.device().index(), jit::mem_stride_copy_gridsize, jit::mem_stride_copy_blocksize, stream->at(0).stream());

        // phase 1. intra-node alltoall
        CHECK_EQ(0, ncclGroupStart());
        for (int g = 0; g < ngpus; g++) {
            CHECK_EQ(0, ncclSend(((char *)output_buff) + g * nnodes * slice_size, nnodes * slice_size, ncclInt8, g + node_rank * ngpus, g_nccl_comm[0], stream->at(0).stream()));
            CHECK_EQ(0, ncclRecv(((char *)input_buff) + g * nnodes * slice_size, nnodes * slice_size, ncclInt8, g + node_rank * ngpus, g_nccl_comm[0], stream->at(0).stream()));
        }
        CHECK_EQ(0, ncclGroupEnd());

        // phase 2. per-gpu (nnodes) stride copy
        slice_size < sizeof(uint4) ? jit::jit_execute({&output_buff, &input_buff, &slice_size, &nnodes, &ngpus}, jit::mem_stride_copy_char_fd,
                                                      input.device().index(), jit::mem_stride_copy_gridsize, jit::mem_stride_copy_blocksize, stream->at(0).stream()) :
                                     jit::jit_execute({&output_buff, &input_buff, &slice_size_uint4, &nnodes, &ngpus}, jit::mem_stride_copy_uint4_fd,
                                                      input.device().index(), jit::mem_stride_copy_gridsize, jit::mem_stride_copy_blocksize, stream->at(0).stream());


        // phase 3. inter-node alltoall
        CHECK_EQ(0, ncclGroupStart());
        for (int n = 0; n < nnodes; n++) {
            CHECK_EQ(0, ncclSend(((char *)output_buff) + n * ngpus * slice_size, ngpus * slice_size, ncclInt8, n * ngpus + local_rank, g_nccl_comm[0], stream->at(0).stream()));
            CHECK_EQ(0, ncclRecv(((char *)input_buff) + n * ngpus * slice_size, ngpus * slice_size, ncclInt8, n * ngpus + local_rank, g_nccl_comm[0], stream->at(0).stream()));
        }
        CHECK_EQ(0, ncclGroupEnd());

        // // Switch to original stream
        // at::cuda::setCurrentCUDAStream(original_stream);
        
        cudaMemcpyAsync(output.data_ptr(), input.data_ptr(), length, cudaMemcpyDeviceToDevice, stream->at(0).stream());
    

        // return input;
    } else {
        CHECK_EQ(0, ncclGroupStart());
        for (int r = 0; r < nranks; r++) {
            CHECK_EQ(0, ncclSend(((char *)input_buff) + r * slice_size, slice_size, ncclInt8, r, g_nccl_comm[0], stream->at(0).stream()));
            CHECK_EQ(0, ncclRecv(((char *)output_buff) + r * slice_size, slice_size, ncclInt8, r, g_nccl_comm[0], stream->at(0).stream()));
        }
        CHECK_EQ(0, ncclGroupEnd());

        // NCCL stream allocator will add blocking event to computation stream after computation kernels
        // c10::cuda::CUDACachingAllocator::recordStream(tmp_output.storage().data_ptr(), original_stream);

        // Switch to original stream
        // at::cuda::setCurrentCUDAStream(original_stream);
    }
}
