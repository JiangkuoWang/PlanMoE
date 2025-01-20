// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <torch/extension.h>

#include "zfp.h"

#if defined(USE_GPU)
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "comm/hetu.h"
#include "comm/naive.h"
#include "comm/pipe.h"
#include "compressor/abstract.h"
#include "compressor/int8.h"
#include "compressor/lz.h"
#include "compressor/no.h"
#include "compressor/zfpc.h"
#include "dd_comm.h"
#include "jit.h"
#else
#undef USE_NCCL
#endif

#if defined(USE_NCCL)
#include <nccl.h>
#endif

#include <regex>
#include <vector>

#if defined(__linux__)
#include <sys/wait.h>
#endif

#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_CPU
#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS

#define CHECK_EQ(x, y) AT_ASSERTM((x) == (y), "CHECK_EQ fails.")
#define CHECK_NE(x, y) AT_ASSERTM((x) != (y), "CHECK_NE fails.")
#define CHECK_LE(x, y) AT_ASSERTM((x) <= (y), "CHECK_LE fails.")
#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

template <typename dtype>
static void invoke_cpu(const std::vector<torch::Tensor> &ts,
                       const std::vector<int> &extra, int kernel_type) {
  int samples = extra[0];
  int hidden = extra[1];
  int capacity = extra[2];
  dtype *gates1_s = static_cast<dtype *>(ts[0].data_ptr());
  int *indices1_s = static_cast<int *>(ts[1].data_ptr());
  int *locations1_s = static_cast<int *>(ts[2].data_ptr());
  dtype *reshaped_input = static_cast<dtype *>(ts[3].data_ptr());
  dtype *dispatched_input = static_cast<dtype *>(ts[4].data_ptr());

  for (int i = 0; i < (int)ts.size(); ++i)
    CHECK_CONTIGUOUS(ts[i]);

  if (kernel_type == 0) { // forward
    for (int i = 0; i < samples; ++i) {
      if (locations1_s[i] < capacity && indices1_s[i] >= 0) {
        for (int j = 0; j < hidden; ++j) {
          dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) *
                               (hidden) +
                           j] += gates1_s[i] * reshaped_input[i * (hidden) + j];
        }
      }
    }
  } else if (kernel_type == 1) { // backward_data
    for (int i = 0; i < samples; ++i) {
      if (locations1_s[i] < capacity && indices1_s[i] >= 0) {
        for (int j = 0; j < hidden; ++j) {
          reshaped_input[i * hidden + j] =
              gates1_s[i] *
              dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) *
                                   (hidden) +
                               j];
        }
      } else {
        for (int j = 0; j < hidden; ++j) {
          reshaped_input[i * hidden + j] = 0;
        }
      }
    }
  } else { // backward_gate
    for (int i = 0; i < samples; ++i) {
      gates1_s[i] = 0;
      if (locations1_s[i] >= capacity || indices1_s[i] < 0)
        continue;
      for (int j = 0; j < hidden; ++j) {
        gates1_s[i] +=
            dispatched_input[(indices1_s[i] * capacity + locations1_s[i]) *
                                 (hidden) +
                             j] *
            reshaped_input[i * hidden + j];
      }
    }
  }
}

#if defined(USE_NCCL)

static std::vector<ncclComm_t> g_nccl_comm;
static std::vector<std::vector<at::cuda::CUDAEvent>> g_cuda_events;
static std::vector<at::cuda::CUDAStream> g_nccl_stream;
static int g_world_size = 0;
static int g_world_rank = 0;
static int g_local_size = 0;
static int g_local_rank = 0;

// jit
extern int mem_stride_copy_char_fd;
extern int mem_stride_copy_uint4_fd;
extern int mem_stride_copy_gridsize;
extern int mem_stride_copy_blocksize;

static size_t get_nccl_unique_id_size() { return sizeof(ncclUniqueId); }

static void get_nccl_unique_id(torch::Tensor &nccl_unique_id_tensor) {
  ncclUniqueId nccl_unique_id;
  int num_stream = nccl_unique_id_tensor.size(0);
  for (int i = 0; i < num_stream; ++i) {
    CHECK_EQ(0, ncclGetUniqueId(&nccl_unique_id));
    // CHECK_CPU(nccl_unique_id_tensor);
    // CHECK_EQ(nccl_unique_id_tensor.nbytes(), sizeof(ncclUniqueId));
    memcpy((void *)nccl_unique_id_tensor.data_ptr() + i * sizeof(ncclUniqueId),
           &nccl_unique_id, sizeof(ncclUniqueId));
  }
}

static void init_nccl(const torch::Tensor &nccl_unique_id_tensor,
                      int world_size, int world_rank, int max_num_split) {
  int num_stream = nccl_unique_id_tensor.size(0);
  ncclUniqueId nccl_unique_id;
  g_nccl_comm.resize(num_stream);
  g_cuda_events.resize(num_stream);
  CHECK_CPU(nccl_unique_id_tensor);
  CHECK_EQ(nccl_unique_id_tensor.nbytes(), num_stream * sizeof(ncclUniqueId));
  for (int i = 0; i < num_stream; ++i) {
    memcpy(&nccl_unique_id,
           ((void *)nccl_unique_id_tensor.data_ptr()) +
               i * sizeof(ncclUniqueId),
           sizeof(ncclUniqueId));
    CHECK_EQ(0, ncclGroupStart());
    CHECK_EQ(0, ncclCommInitRank(&g_nccl_comm[i], world_size, nccl_unique_id,
                                 world_rank));
    CHECK_EQ(0, ncclGroupEnd());
    g_nccl_stream.emplace_back(at::cuda::getStreamFromPool());
    g_cuda_events[i].resize(max_num_split);
  }

  g_world_size = world_size;
  g_world_rank = world_rank;

  if (const char *local_size = std::getenv("LOCAL_SIZE")) {
    g_local_size = std::atoi(local_size);
  } else {
    CHECK_EQ(0, cudaGetDeviceCount(&g_local_size));
  }
  CHECK_EQ(0, ncclCommCuDevice(g_nccl_comm[0], &g_local_rank));
  // jit for nccl
  jit::jit_init(g_local_rank);
}

static torch::Tensor &current_stream_release(torch::Tensor &tensor, int idx) {
  return tensor;
}

static torch::Tensor &current_stream_acquire(torch::Tensor &tensor, int idx) {
  return tensor;
}

static torch::Tensor &nccl_stream_release(torch::Tensor &tensor, int idx) {
  return tensor;
}

static torch::Tensor &nccl_stream_acquire(torch::Tensor &tensor, int idx) {
  return tensor;
}

static AbstractCompressor *
get_compressor(const std::string &name,
               std::shared_ptr<AbstractComm> comm_ptr) {
  if (name == "int8") {
    return new Int8Compressor(comm_ptr);
  }
  if (name == "zfp") {
    return new ZfpCompressor(comm_ptr);
  }
  if (name == "lz") {
    return new LzCompressor(comm_ptr);
  }
  return new NoCompressor(comm_ptr);
}

static std::shared_ptr<AbstractComm> get_comm(const std::string &name) {
  if (name == "dd") {
    return std::make_shared<DdComm>(&g_nccl_stream, g_nccl_comm, g_world_size,
                                    g_world_rank, g_local_size, g_local_rank);
  }
  if (name == "pipe") {
    return std::make_shared<PipeComm>(&g_nccl_stream, g_nccl_comm, g_world_size,
                                      g_world_rank, g_local_size, g_local_rank);
  }
  if (name == "hetu") {
    return std::make_shared<HeTuComm>(&g_nccl_stream, g_nccl_comm, g_world_size,
                                      g_world_rank, g_local_size, g_local_rank);
  }
  return std::make_shared<NaiveComm>(&g_nccl_stream, g_nccl_comm, g_world_size,
                                     g_world_rank, g_local_size, g_local_rank);
}

static std::vector<AbstractCompressor *> compress_ptr_lst;
static int comm_cnt = 0;
static int decompress_cnt = 0;

static void clear_ptr_lst() {
  for (auto &compress_ptr : compress_ptr_lst) {
    if (compress_ptr) {
      delete compress_ptr;
      compress_ptr = nullptr;
    }
  }
  compress_ptr_lst.resize(0);
  comm_cnt = 0;
  decompress_cnt = 0;
}

static torch::Tensor compress_operation(const torch::Tensor &input,
                                        const std::string &str,
                                        const std::string &comm_name) {
  size_t idx = compress_ptr_lst.size();
  std::shared_ptr<AbstractComm> comm_ptr = get_comm(comm_name);
  // std::make_shared<NaiveComm>(&g_nccl_stream, g_nccl_comm, g_world_size,
  //                             g_world_rank, g_local_size, g_local_rank);
  compress_ptr_lst.emplace_back(get_compressor(str, comm_ptr));
  torch::Tensor after_compress = compress_ptr_lst.back()->compress(input);
  for (auto &events : g_cuda_events) {
    events[idx].record(at::cuda::getCurrentCUDAStream());
  }
  for (auto &nccl_stream : g_nccl_stream) {
    c10::cuda::CUDACachingAllocator::recordStream(
        after_compress.storage().data_ptr(), nccl_stream);
  }
  return after_compress;
}

static torch::Tensor comm_operation(const torch::Tensor &input) {
  const int idx = comm_cnt++;
  const at::cuda::CUDAStream &original_stream =
      at::cuda::getCurrentCUDAStream();
  // std::cout << "????" << std::endl;
  compress_ptr_lst[idx]->pre_comm(&original_stream);

  // std::cout << "!!!!" << std::endl;
  for (int i = 0; i < g_nccl_stream.size(); ++i) {
    g_cuda_events[i][idx].block(g_nccl_stream[i]);
  }
  compress_ptr_lst[idx]->all_to_all(input, compress_ptr_lst[idx]->g_output);
  for (int i = 0; i < g_nccl_stream.size(); ++i) {
    g_cuda_events[i][idx].record(g_nccl_stream[i]);
  }
  return compress_ptr_lst[idx]->g_output;
}

static torch::Tensor decompress_operation(const torch::Tensor &input) {
  const int idx = decompress_cnt++;
  for (auto &event : g_cuda_events) {
    event[idx].block(at::cuda::getCurrentCUDAStream());
  }
  torch::Tensor output = compress_ptr_lst[idx]->decompress(input);
  delete compress_ptr_lst[idx];
  compress_ptr_lst[idx] = nullptr;
  return output;
}

#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#if defined(USE_GPU)

  m.def("update_sdk_home", &jit::update_sdk_home,
        "Configure SDK HOME Path for GPU (CUDA)");
  m.def("invoke", &jit::invoke, "Generic Invoke for GPU (CUDA)");
  m.def("inject_source", &jit::inject_source, "Inject Source for GPU (CUDA)");
#endif
  m.def("invoke_cpu_fp32", &invoke_cpu<float>, "Invoke for Sparse Ops (CPU)");
  m.def("invoke_cpu_fp64", &invoke_cpu<double>, "Invoke for Sparse Ops (CPU)");
#if defined(USE_NCCL)
  m.def("get_nccl_unique_id_size", &get_nccl_unique_id_size,
        "Get size of ncclUniqueId in bytes");
  m.def("get_nccl_unique_id", &get_nccl_unique_id,
        "Get ncclUniqueId for NCCL initialization");
  m.def("init_nccl", &init_nccl, "NCCL initialization");
  m.def("current_stream_release", &current_stream_release,
        "Record CUDA event on current stream to i-th event slot");
  m.def("current_stream_acquire", &current_stream_acquire,
        "Let current stream wait CUDA event in i-th event slot");
  m.def("nccl_stream_release", &nccl_stream_release,
        "Record CUDA event on NCCL stream to i-th event slot");
  m.def("nccl_stream_acquire", &nccl_stream_acquire,
        "Let NCCL stream wait CUDA event in i-th event slot");
  m.def("compress_operation", &compress_operation, "Compress Operation");
  m.def("comm_operation", &comm_operation, "Comm Operation");
  m.def("decompress_operation", &decompress_operation, "Decompress Operation");
  m.def("clear_ptr_lst", &clear_ptr_lst, "Clear Ptr Lst");
#endif
}

#if defined(USE_GPU)
#include <torch/script.h>
#define DEFINE_KERNEL(x, y)                                                    \
  static int x = -1;                                                           \
  if (x == -1) {                                                               \
    x = y;                                                                     \
  }

torch::Tensor warp_cumsum(torch::Tensor x) {
  CHECK_CUDA(x);
  CHECK_EQ(x.dim(), 2);
  x = x.to(torch::kInt32).contiguous();

  auto y = torch::empty_like(x);

  DEFINE_KERNEL(cumsum_fn, jit::inject_source(R"(
extern "C" __global__ void cumsum_fn(int* input0 /* (num_samples, batch_num) */, int* output0 /* (num_samples, batch_num) */, int num_samples) {
    #define thread_num  1024
    #define batch_num ((int)gridDim.x)

    __shared__ int temp[thread_num + 1];
    int thid = threadIdx.x, bid = blockIdx.x;
    int last_sum = -1;

    for (int S = 0; S < num_samples; S += thread_num, output0 += thread_num * batch_num, input0 += thread_num * batch_num) {
        int offset = 1;
        if (S + thid < num_samples)
                temp[thid] = input0[thid * batch_num + bid];
        for (int d = thread_num >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d)
                        temp[offset * (2 * thid + 2) - 1] += temp[offset * (2 * thid + 1) - 1];
                offset *= 2;
        }
        if (thid == 0)
                temp[thread_num] = temp[thread_num - 1], temp[thread_num - 1] = 0;
        for (int d = 1; d < thread_num; d *= 2) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                        int ai = offset * (2 * thid + 1) - 1;
                        int bi = offset * (2 * thid + 2) - 1;
                        int t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                }
        }
        __syncthreads();
        if (S + thid < num_samples)
                output0[thid * batch_num + bid] = temp[thid + 1] + last_sum;
        __syncthreads();
        last_sum += temp[thread_num];
    }
}
)"));

  jit::jit_execute_with_values({x.data_ptr(), y.data_ptr(), (void *)x.size(0)},
                               cumsum_fn, x.device().index(), x.size(1), 1024,
                               nullptr);
  return y;
}

TORCH_LIBRARY(tutel_ops, m) { m.def("cumsum", warp_cumsum); }
#endif
