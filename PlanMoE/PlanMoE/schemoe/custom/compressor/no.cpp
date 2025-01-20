#include "no.h"

torch::Tensor NoCompressor::compress(const torch::Tensor &input) {
    g_output = at::empty_like(input);
    for (auto &nccl_stream : *comm_ptr->stream) {
        c10::cuda::CUDACachingAllocator::recordStream(g_output.storage().data_ptr(), nccl_stream);
    }
    return input;
}
torch::Tensor NoCompressor::decompress(const torch::Tensor &input) {
    return input;
}

NoCompressor::NoCompressor(std::shared_ptr<AbstractComm> comm_ptr) :
    AbstractCompressor(comm_ptr) {
}

void NoCompressor::all_to_all(const torch::Tensor &input, const torch::Tensor &output) {
    comm_ptr->all_to_all(input, output, input.nbytes());
}