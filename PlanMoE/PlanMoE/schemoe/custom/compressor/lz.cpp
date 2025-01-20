#include "lz.h"

torch::Tensor LzCompressor::compress(const torch::Tensor &input) {
    std::vector<at::Tensor> input_list = at::split(input, input.size(0) / comm_ptr->g_world_size, 0);
    std::vector<at::Tensor> flagArrOffsetGlobalList;
    std::vector<at::Tensor> compressedDataOffsetGlobalList;
    std::vector<at::Tensor> flagArrGlobalList;
    std::vector<at::Tensor> compressedDataGlobalList;
    for (auto &slice : input_list) {
        std::vector<torch::Tensor> compress_ret = lz_compress(slice, at::cuda::getCurrentCUDAStream());
        // placeholderList.push_back(compress_ret[0]);
        flagArrOffsetGlobalList.push_back(compress_ret[0]);
        compressedDataOffsetGlobalList.push_back(compress_ret[1]);
        flagArrGlobalList.push_back(compress_ret[2]);
        compressedDataGlobalList.push_back(compress_ret[3]);
    }
    flagArrOffsetGlobal          = torch::cat(flagArrOffsetGlobalList, 0).contiguous();
    compressedDataOffsetGlobal   = torch::cat(compressedDataOffsetGlobalList, 0).contiguous();
    flagArrGlobal                = torch::cat(flagArrGlobalList, 0).contiguous();
    compressedDataGlobal         = torch::cat(compressedDataGlobalList, 0).contiguous();
    g_flagArrOffsetGlobal        = at::empty_like(flagArrOffsetGlobal);
    g_compressedDataOffsetGlobal = at::empty_like(compressedDataOffsetGlobal);
    g_flagArrGlobal              = at::empty_like(flagArrGlobal);
    g_compressedDataGlobal       = at::empty_like(compressedDataGlobal);
    g_output                     = at::zeros_like(input);

    for (auto &nccl_stream : *comm_ptr->stream) {
        c10::cuda::CUDACachingAllocator::recordStream(flagArrOffsetGlobal.storage().data_ptr(), nccl_stream);
        c10::cuda::CUDACachingAllocator::recordStream(compressedDataOffsetGlobal.storage().data_ptr(), nccl_stream);
        c10::cuda::CUDACachingAllocator::recordStream(flagArrGlobal.storage().data_ptr(), nccl_stream);
        c10::cuda::CUDACachingAllocator::recordStream(compressedDataGlobal.storage().data_ptr(), nccl_stream);
        c10::cuda::CUDACachingAllocator::recordStream(g_flagArrOffsetGlobal.storage().data_ptr(), nccl_stream);
        c10::cuda::CUDACachingAllocator::recordStream(g_compressedDataOffsetGlobal.storage().data_ptr(), nccl_stream);
        c10::cuda::CUDACachingAllocator::recordStream(g_flagArrGlobal.storage().data_ptr(), nccl_stream);
        c10::cuda::CUDACachingAllocator::recordStream(g_compressedDataGlobal.storage().data_ptr(), nccl_stream);
    }

    // for (auto &nccl_stream : *comm_ptr->stream) {
    //     c10::cuda::CUDACachingAllocator::recordStream(placeholder.storage().data_ptr(), nccl_stream);
    // }
    return input;
}
torch::Tensor LzCompressor::decompress(const torch::Tensor &input) {
    int                     g_world_size                   = comm_ptr->g_world_size;
    std::vector<at::Tensor> input_list                     = at::split(input, input.size(0) / g_world_size, 0);
    std::vector<at::Tensor> flagArrOffsetGlobalList        = at::split(g_flagArrOffsetGlobal, g_flagArrOffsetGlobal.size(0) / g_world_size, 0);
    std::vector<at::Tensor> compressedDataOffsetGlobalList = at::split(g_compressedDataOffsetGlobal, g_compressedDataOffsetGlobal.size(0) / g_world_size, 0);
    std::vector<at::Tensor> flagArrGlobalList              = at::split(g_flagArrGlobal, g_flagArrGlobal.size(0) / g_world_size, 0);
    std::vector<at::Tensor> compressedDataGlobalList       = at::split(g_compressedDataGlobal, g_compressedDataGlobal.size(0) / g_world_size, 0);
    int                     numOfBlocks                    = flagArrOffsetGlobalList[0].size(0) - 1;
    // std::cout << numOfBlocks << std::endl;
    for (int i = 0; i < g_world_size; ++i) {
        lz_decompress(input_list[i],
                      numOfBlocks,
                      flagArrOffsetGlobalList[i],
                      compressedDataOffsetGlobalList[i],
                      flagArrGlobalList[i],
                      compressedDataGlobalList[i],
                      at::cuda::getCurrentCUDAStream());
    }
    return input;
}

LzCompressor::LzCompressor(std::shared_ptr<AbstractComm> comm_ptr) :
    AbstractCompressor(comm_ptr) {
}

void LzCompressor::all_to_all(const torch::Tensor &input, const torch::Tensor &output) {
    // comm_ptr->all_to_all(input, output, input.nbytes());
    comm_ptr->all_to_all(flagArrOffsetGlobal, g_flagArrOffsetGlobal, flagArrOffsetGlobal.nbytes());
    comm_ptr->all_to_all(compressedDataOffsetGlobal, g_compressedDataOffsetGlobal, compressedDataOffsetGlobal.nbytes());
    comm_ptr->all_to_all(flagArrGlobal, g_flagArrGlobal, flagArrGlobal.nbytes());
    comm_ptr->all_to_all(compressedDataGlobal, g_compressedDataGlobal, compressedDataGlobal.nbytes());
}