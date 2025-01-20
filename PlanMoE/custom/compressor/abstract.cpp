#include "abstract.h"

AbstractCompressor::AbstractCompressor(std::shared_ptr<AbstractComm> comm_ptr) {
    this->comm_ptr = comm_ptr;
}

AbstractCompressor::~AbstractCompressor() {
    comm_ptr.reset();
}

void AbstractCompressor::pre_comm(const at::cuda::CUDAStream *cal_stream) {
}