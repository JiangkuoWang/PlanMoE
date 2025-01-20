#include "zfpc.h"

ZfpCompressor::ZfpCompressor(std::shared_ptr<AbstractComm> comm_ptr,
                             double                        compress_rate) noexcept
    :
    AbstractCompressor(comm_ptr) {
    this->compress_rate = compress_rate;
    this->buffer        = nullptr;
    this->last_bufsize  = 0;
}

ZfpCompressor::~ZfpCompressor() {
    if (buffer) {
        // cudaFreeAsync(buffer, at::cuda::getCurrentCUDAStream().stream());
        buffer = nullptr;
    }
};

void ZfpCompressor::set_compress_rate(const double &compress_rate) {
    this->compress_rate = compress_rate;
}

torch::Tensor ZfpCompressor::compress(const torch::Tensor &input) {
    sizes = input.sizes().vec();
    // std::cout << sizes[0] << " " << sizes[1] << " " << sizes[2] << std::endl;
    AT_ASSERTM(sizes[0] % 4 == 0, "zfp fails.");
    AT_ASSERTM(sizes[1] % 4 == 0, "zfp fails.");
    AT_ASSERTM(sizes[2] % 4 == 0, "zfp fails.");
    // std::cout << sizes[0] % 4 << " " << sizes[1] % 4 << " " << sizes[2] % 4
    //           << std::endl;
    zfp_type     type        = zfp_type_float; /* array scalar type */
    zfp_field   *field       = zfp_field_2d((char *)input.data_ptr(), type, sizes[2],
                                            sizes[1] * sizes[0]);
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    field->cuda_stream       = (void *)&cuda_stream;
    zfp_stream *zfp          = zfp_stream_open(NULL);
    zfp_stream_set_rate(zfp, compress_rate, type, 2, 0);
    zfp_stream_set_execution(zfp, zfp_exec_cuda);
    size_t bufsize = zfp_stream_maximum_size(zfp, field);

    torch::Tensor output = torch::empty(
        {(bufsize + 1) / 2},
        torch::TensorOptions().device(input.device()).dtype(at::kHalf),
        torch::MemoryFormat::Contiguous);

    bitstream *stream = stream_open(output.data_ptr(), bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    length = zfp_compress(zfp, field);
    // std::cout << length << std::endl;
    zfp_field_free(field);
    zfp_stream_close(zfp);

    g_output = torch::empty_like(output);
    for (auto &nccl_stream : *comm_ptr->stream) {
        c10::cuda::CUDACachingAllocator::recordStream(
            g_output.storage().data_ptr(), nccl_stream);
    }
    /*output = at::reshape(output, {2, 3, 16});
    output = at::permute(output, {1, 0, 2});
    output = output.contiguous();*/

    return output;
}

torch::Tensor ZfpCompressor::decompress(const torch::Tensor &input) {
    zfp_type type = zfp_type_float; /* array scalar type */
    // std::cout << sizes;
    torch::Tensor output = torch::empty(
        sizes, torch::TensorOptions().device(input.device()).dtype(at::kFloat),
        torch::MemoryFormat::Contiguous);
    // std::cout << sizes;
    zfp_field   *field       = zfp_field_2d((char *)output.data_ptr(), type, sizes[2],
                                            sizes[1] * sizes[0]);
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    field->cuda_stream       = (void *)&cuda_stream;
    zfp_stream *zfp          = zfp_stream_open(NULL);
    zfp_stream_set_rate(zfp, compress_rate, type, 2, 0);
    zfp_stream_set_execution(zfp, zfp_exec_cuda);
    size_t     bufsize = zfp_stream_maximum_size(zfp, field);
    bitstream *stream  = stream_open(input.data_ptr(), bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_decompress(zfp, field);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    return output;
}

void ZfpCompressor::all_to_all(const torch::Tensor &input,
                               const torch::Tensor &output) {
    // cudaMemcpyAsync(output.data_ptr(), input.data_ptr(), input.nbytes(),
    // cudaMemcpyDeviceToDevice, stream->stream());

    // std::cout <<length<<" ";
    comm_ptr->all_to_all(input, output, length);
}