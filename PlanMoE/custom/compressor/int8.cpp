#include "int8.h"

Int8Compressor::Int8Compressor(std::shared_ptr<AbstractComm> comm_ptr) :
    AbstractCompressor(comm_ptr) {
}

torch::Tensor Int8Compressor::compress(const torch::Tensor &input) {
    sizes                = input.sizes().vec();
    dtype                = input.dtype();
    bias                 = std::get<0>(torch::min(input, -1, true));
    scale                = std::get<0>(torch::max(input, -1, true));
    scale                = at::sub(scale, bias);
    torch::Tensor output = at::sub(input, bias);
    output               = at::div(output, scale);
    output               = at::mul(output, 255.0);
    output               = output.to(torch::kUInt8);
    length               = output.nbytes();

    torch::Tensor fp_output = torch::empty(
        {(length + 3) / 4},
        torch::TensorOptions().device(input.device()).dtype(at::kFloat),
        torch::MemoryFormat::Contiguous);
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemcpyAsync(fp_output.data_ptr(), output.data_ptr(), length, cudaMemcpyDeviceToDevice, cuda_stream);
    for (auto &nccl_stream : *comm_ptr->stream) {
        c10::cuda::CUDACachingAllocator::recordStream(bias.storage().data_ptr(), nccl_stream);
    }
    for (auto &nccl_stream : *comm_ptr->stream) {
        c10::cuda::CUDACachingAllocator::recordStream(scale.storage().data_ptr(), nccl_stream);
    }
    g_bias  = at::empty_like(bias);
    g_scale = at::empty_like(scale);
    for (auto &nccl_stream : *comm_ptr->stream) {
        c10::cuda::CUDACachingAllocator::recordStream(g_bias.storage().data_ptr(), nccl_stream);
    }
    for (auto &nccl_stream : *comm_ptr->stream) {
        c10::cuda::CUDACachingAllocator::recordStream(g_scale.storage().data_ptr(), nccl_stream);
    }
    this->g_output = at::empty_like(fp_output);
    for (auto &nccl_stream : *comm_ptr->stream) {
        c10::cuda::CUDACachingAllocator::recordStream(g_output.storage().data_ptr(), nccl_stream);
    }
    return fp_output;
}

torch::Tensor Int8Compressor::decompress(const torch::Tensor &input) {
    torch::Tensor output = torch::empty(
        sizes,
        torch::TensorOptions().device(input.device()).dtype(torch::kUInt8),
        torch::MemoryFormat::Contiguous);
    // std::cout << g_bias;
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    cudaMemcpyAsync(output.data_ptr(), input.data_ptr(), length, cudaMemcpyDeviceToDevice, cuda_stream);

    output = output.to(dtype);
    output = at::div(output, 255.0);
    output = at::mul(output, g_scale);
    output = at::add(output, g_bias);
    return output;
}

void Int8Compressor::pre_comm(const at::cuda::CUDAStream *cal_stream) {
    // g_bias  = at::empty_like(bias);
    // g_scale = at::empty_like(scale);
    // c10::cuda::CUDACachingAllocator::recordStream(g_scale.storage().data_ptr(), *cal_stream);
    // c10::cuda::CUDACachingAllocator::recordStream(g_bias.storage().data_ptr(), *cal_stream);
}

void Int8Compressor::all_to_all(const torch::Tensor &input, const torch::Tensor &output) {
    // std::cout << (output).data_ptr() << std::endl;
    comm_ptr->all_to_all(input, output, length);
    comm_ptr->all_to_all(bias, g_bias, bias.nbytes());
    comm_ptr->all_to_all(scale, g_scale, scale.nbytes());
}