#include <cuda.h>
#include <cuda_runtime.h>
// #include <thrust/system/cuda/detail/cub/device/device_scan.cuh>
#include <cub/cub.cuh>
#include <torch/extension.h>
#define BLOCK_SIZE 2048 // in unit of byte, the size of one data block
#define THREAD_SIZE                                                            \
  128 // in unit of datatype, the size of the thread block, so as the size of
      // symbols per iteration
#define WINDOW_SIZE                                                            \
  32 // in unit of datatype, maximum 255, the size of the sliding window, so as
     // the maximum match length
#define INPUT_TYPE float

// Define the compress match kernel functions
template <typename T>
__global__ void
compressKernelI(T *input, uint32_t numOfBlocks, int *flagArrSizeGlobal,
                int *compressedDataSizeGlobal, uint8_t *tmpFlagArrGlobal,
                uint8_t *tmpCompressedDataGlobal, int minEncodeLength) {
  // Block size in uint of datatype
  const uint32_t blockSize = BLOCK_SIZE / sizeof(T);

  // Window size in uint of datatype
  const uint32_t threadSize = THREAD_SIZE;

  // Allocate shared memory for the lookahead buffer of the whole block, the
  // sliding window is included
  __shared__ T buffer[blockSize];
  __shared__ uint8_t lengthBuffer[blockSize];
  __shared__ uint8_t offsetBuffer[blockSize];
  __shared__ uint32_t prefixBuffer[blockSize + 1];

  // initialize the tid
  int tid = 0;

  // Copy the memeory from global to shared
  for (int i = 0; i < blockSize / threadSize; i++) {
    buffer[threadIdx.x + threadSize * i] =
        input[blockIdx.x * blockSize + threadIdx.x + threadSize * i];
  }

  // Synchronize all threads to ensure that the buffer is fully loaded
  __syncthreads();

  // find match for every data point
  for (int iteration = 0; iteration < (int)(blockSize / threadSize);
       iteration++) {
    // Initialize the lookahead buffer and the sliding window pointers
    tid = threadIdx.x + iteration * threadSize;
    int bufferStart = tid;
    int bufferPointer = bufferStart;
    int windowStart =
        bufferStart - int(WINDOW_SIZE) < 0 ? 0 : bufferStart - WINDOW_SIZE;
    int windowPointer = windowStart;

    uint8_t maxLen = 0;
    uint8_t maxOffset = 0;
    uint8_t len = 0;
    uint8_t offset = 0;

    while (windowPointer < bufferStart && bufferPointer < blockSize) {
      if (buffer[bufferPointer] == buffer[windowPointer]) {
        if (offset == 0) {
          offset = bufferPointer - windowPointer;
        }
        len++;
        bufferPointer++;
      } else {
        if (len > maxLen) {
          maxLen = len;
          maxOffset = offset;
        }
        len = 0;
        offset = 0;
        bufferPointer = bufferStart;
      }
      windowPointer++;
    }
    if (len > maxLen) {
      maxLen = len;
      maxOffset = offset;
    }

    lengthBuffer[threadIdx.x + iteration * threadSize] = maxLen;
    offsetBuffer[threadIdx.x + iteration * threadSize] = maxOffset;

    // initialize array as 0
    prefixBuffer[threadIdx.x + iteration * threadSize] = 0;
  }
  __syncthreads();

  // find encode information
  uint32_t flagCount = 0;
  __shared__ uint8_t byteFlagArr[(blockSize / 8)];

  if (threadIdx.x == 0) {
    uint8_t flagPosition = 0x01;
    uint8_t byteFlag = 0;

    int encodeIndex = 0;

    while (encodeIndex < blockSize) {
      // if length < minEncodeLength, no match is found
      if (lengthBuffer[encodeIndex] < minEncodeLength) {
        prefixBuffer[encodeIndex] = sizeof(T);
        encodeIndex++;
      }
      // if length > minEncodeLength, match is found
      else {
        prefixBuffer[encodeIndex] = 2;
        encodeIndex += lengthBuffer[encodeIndex];
        byteFlag |= flagPosition;
      }
      // store the flag if there are 8 bits already
      if (flagPosition == 0x80) {
        byteFlagArr[flagCount] = byteFlag;
        flagCount++;
        flagPosition = 0x01;
        byteFlag = 0;
        continue;
      }
      flagPosition <<= 1;
    }
    if (flagPosition != 0x01) {
      byteFlagArr[flagCount] = byteFlag;
      flagCount++;
    }
  }
  __syncthreads();

  // prefix summation, up-sweep
  int prefixSumOffset = 1;
  for (uint32_t d = blockSize >> 1; d > 0; d = d >> 1) {
    for (int iteration = 0; iteration < (int)(blockSize / threadSize);
         iteration++) {
      tid = threadIdx.x + iteration * threadSize;
      if (tid < d) {
        int ai = prefixSumOffset * (2 * tid + 1) - 1;
        int bi = prefixSumOffset * (2 * tid + 2) - 1;
        prefixBuffer[bi] += prefixBuffer[ai];
      }
      __syncthreads();
    }
    prefixSumOffset *= 2;
  }

  // clear the last element
  if (threadIdx.x == 0) {
    // printf("block size: %d flag array size: %d\n", prefixBuffer[blockSize -
    // 1], flagCount);
    compressedDataSizeGlobal[blockIdx.x] = prefixBuffer[blockSize - 1];
    flagArrSizeGlobal[blockIdx.x] = flagCount;
    prefixBuffer[blockSize] = prefixBuffer[blockSize - 1];
    prefixBuffer[blockSize - 1] = 0;
  }
  __syncthreads();

  // prefix summation, down-sweep
  for (int d = 1; d < blockSize; d *= 2) {
    prefixSumOffset >>= 1;
    for (int iteration = 0; iteration < (int)(blockSize / threadSize);
         iteration++) {
      tid = threadIdx.x + iteration * threadSize;

      if (tid < d) {
        int ai = prefixSumOffset * (2 * tid + 1) - 1;
        int bi = prefixSumOffset * (2 * tid + 2) - 1;

        uint32_t t = prefixBuffer[ai];
        prefixBuffer[ai] = prefixBuffer[bi];
        prefixBuffer[bi] += t;
      }
      __syncthreads();
    }
  }

  // encoding phase one
  int tmpCompressedDataGlobalOffset;
  tmpCompressedDataGlobalOffset = blockSize * blockIdx.x * sizeof(T);
  for (int iteration = 0; iteration < (int)(blockSize / threadSize);
       iteration++) {
    tid = threadIdx.x + iteration * threadSize;
    if (prefixBuffer[tid + 1] != prefixBuffer[tid]) {
      if (lengthBuffer[tid] < minEncodeLength) {
        uint32_t tmpOffset = prefixBuffer[tid];
        uint8_t *bytePtr = (uint8_t *)&buffer[tid];
        for (int tmpIndex = 0; tmpIndex < sizeof(T); tmpIndex++) {
          tmpCompressedDataGlobal[tmpCompressedDataGlobalOffset + tmpOffset +
                                  tmpIndex] = *(bytePtr + tmpIndex);
        }
      } else {
        uint32_t tmpOffset = prefixBuffer[tid];
        tmpCompressedDataGlobal[tmpCompressedDataGlobalOffset + tmpOffset] =
            lengthBuffer[tid];
        tmpCompressedDataGlobal[tmpCompressedDataGlobalOffset + tmpOffset + 1] =
            offsetBuffer[tid];
      }
    }
  }

  // Copy the memeory back
  if (threadIdx.x == 0) {
    for (int flagArrIndex = 0; flagArrIndex < flagCount; flagArrIndex++) {
      tmpFlagArrGlobal[blockSize / 8 * blockIdx.x + flagArrIndex] =
          byteFlagArr[flagArrIndex];
    }
  }
}

// Define the compress Encode kernel functions
template <typename T>
__global__ void
compressKernelIII(uint32_t numOfBlocks, int *flagArrOffsetGlobal,
                  int *compressedDataOffsetGlobal, uint8_t *tmpFlagArrGlobal,
                  uint8_t *tmpCompressedDataGlobal, uint8_t *flagArrGlobal,
                  uint8_t *compressedDataGlobal) {
  // Block size in uint of bytes
  const int blockSize = BLOCK_SIZE / sizeof(T);

  // Window size in uint of bytes
  const int threadSize = THREAD_SIZE;

  // find block index
  int blockIndex = blockIdx.x;

  int flagArrOffset = flagArrOffsetGlobal[blockIndex];
  int flagArrSize =
      flagArrOffsetGlobal[blockIndex + 1] - flagArrOffsetGlobal[blockIndex];

  int compressedDataOffset = compressedDataOffsetGlobal[blockIndex];
  int compressedDataSize = compressedDataOffsetGlobal[blockIndex + 1] -
                           compressedDataOffsetGlobal[blockIndex];

  int tid = threadIdx.x;

  while (tid < flagArrSize) {
    flagArrGlobal[flagArrOffset + tid] =
        tmpFlagArrGlobal[blockSize / 8 * blockIndex + tid];
    tid += threadSize;
  }

  tid = threadIdx.x;

  while (tid < compressedDataSize) {
    compressedDataGlobal[compressedDataOffset + tid] =
        tmpCompressedDataGlobal[blockSize * sizeof(T) * blockIndex + tid];
    tid += threadSize;
  }
}

// Define the decompress kernel functions
template <typename T>
__global__ void
decompressKernel(T *output, uint32_t numOfBlocks, int *flagArrOffsetGlobal,
                 int *compressedDataOffsetGlobal, uint8_t *flagArrGlobal,
                 uint8_t *compressedDataGlobal) {
  // Block size in unit of datatype
  const uint32_t blockSize = BLOCK_SIZE / sizeof(T);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < numOfBlocks) {
    int flagArrOffset = flagArrOffsetGlobal[tid];
    int flagArrSize = flagArrOffsetGlobal[tid + 1] - flagArrOffsetGlobal[tid];

    int compressedDataOffset = compressedDataOffsetGlobal[tid];

    uint32_t dataPointsIndex = 0;
    uint32_t compressedDataIndex = 0;

    uint8_t byteFlag;

    for (int flagArrayIndex = 0; flagArrayIndex < flagArrSize;
         flagArrayIndex++) {
      byteFlag = flagArrGlobal[flagArrOffset + flagArrayIndex];

      for (int bitCount = 0; bitCount < 8; bitCount++) {
        int matchFlag = (byteFlag >> bitCount) & 0x1;
        if (matchFlag == 1) {
          int length =
              compressedDataGlobal[compressedDataOffset + compressedDataIndex];
          int offset = compressedDataGlobal[compressedDataOffset +
                                            compressedDataIndex + 1];
          compressedDataIndex += 2;
          int dataPointsStart = dataPointsIndex;
          for (int tmpDecompIndex = 0; tmpDecompIndex < length;
               tmpDecompIndex++) {
            output[tid * blockSize + dataPointsIndex] =
                output[tid * blockSize + dataPointsStart - offset +
                       tmpDecompIndex];
            dataPointsIndex++;
          }
        } else {
          uint8_t *tmpPtr =
              (uint8_t *)&output[tid * blockSize + dataPointsIndex];
          for (int tmpDecompIndex = 0; tmpDecompIndex < sizeof(T);
               tmpDecompIndex++) {
            *(tmpPtr + tmpDecompIndex) =
                compressedDataGlobal[compressedDataOffset +
                                     compressedDataIndex + tmpDecompIndex];
          }

          compressedDataIndex += sizeof(T);
          dataPointsIndex++;
        }
        if (dataPointsIndex >= blockSize) {
          return;
        }
      }
    }
  }
}

// torch::Tensor gpu_lz_cuda(
//     torch::Tensor input,
//     cudaStream_t  stream) {
//     uint32_t fileSize = input.nbytes();
//     // calculate the padding size, unit in bytes
//     uint32_t paddingSize = (BLOCK_SIZE - fileSize % BLOCK_SIZE) % BLOCK_SIZE;
//     // calculate the datatype size, unit in datatype
//     uint32_t datatypeSize = (fileSize + paddingSize) / sizeof(INPUT_TYPE);
//     uint32_t numOfBlocks  = (fileSize + paddingSize) / BLOCK_SIZE;
//     // std::cout << fileSize << " " << datatypeSize << " " << numOfBlocks;
//     uint32_t *flagArrSizeGlobal;
//     uint32_t *flagArrOffsetGlobal;
//     uint32_t *compressedDataSizeGlobal;
//     uint32_t *compressedDataOffsetGlobal;
//     uint8_t  *tmpFlagArrGlobal;
//     uint8_t  *tmpCompressedDataGlobal;
//     uint8_t  *flagArrGlobal;
//     uint8_t  *compressedDataGlobal;

//     cudaMalloc((void **)&flagArrSizeGlobal, sizeof(uint32_t) * (numOfBlocks +
//     1)); cudaMalloc((void **)&flagArrOffsetGlobal, sizeof(uint32_t) *
//     (numOfBlocks + 1)); cudaMalloc((void **)&compressedDataSizeGlobal,
//     sizeof(uint32_t) * (numOfBlocks + 1)); cudaMalloc((void
//     **)&compressedDataOffsetGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
//     cudaMalloc((void **)&tmpFlagArrGlobal, sizeof(uint8_t) * datatypeSize /
//     8); cudaMalloc((void **)&tmpCompressedDataGlobal, fileSize +
//     paddingSize); cudaMalloc((void **)&flagArrGlobal, sizeof(uint8_t) *
//     datatypeSize / 8); cudaMalloc((void **)&compressedDataGlobal, fileSize +
//     paddingSize);

//     cudaMemset(flagArrSizeGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
//     cudaMemset(flagArrOffsetGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
//     cudaMemset(compressedDataSizeGlobal, 0, sizeof(uint32_t) * (numOfBlocks +
//     1)); cudaMemset(compressedDataOffsetGlobal, 0, sizeof(uint32_t) *
//     (numOfBlocks + 1)); cudaMemset(tmpFlagArrGlobal, 0, sizeof(uint8_t) *
//     datatypeSize / 8); cudaMemset(tmpCompressedDataGlobal, 0,
//     sizeof(INPUT_TYPE) * datatypeSize); torch::Tensor output =
//     torch::ones(fileSize + paddingSize,
//     torch::TensorOptions().device(input.device()).dtype(input.dtype()));

//     dim3 gridDim(numOfBlocks);
//     dim3 blockDim(THREAD_SIZE);
//     dim3 deGridDim((numOfBlocks + 31) / 32);
//     dim3 deBlockDim(32);

//     int minEncodeLength = sizeof(INPUT_TYPE) == 1 ? 2 : 1;

//     // launch kernels
//     compressKernelI<INPUT_TYPE><<<gridDim, blockDim, 0,
//     stream>>>(input.data_ptr<INPUT_TYPE>(), numOfBlocks, flagArrSizeGlobal,
//     compressedDataSizeGlobal, tmpFlagArrGlobal, tmpCompressedDataGlobal,
//     minEncodeLength);

//     // Determine temporary device storage requirements
//     void  *flag_d_temp_storage     = NULL;
//     size_t flag_temp_storage_bytes = 0;
//     cub::DeviceScan::ExclusiveSum(flag_d_temp_storage,
//     flag_temp_storage_bytes, flagArrSizeGlobal, flagArrOffsetGlobal,
//     numOfBlocks + 1);
//     // Allocate temporary storage
//     cudaMalloc(&flag_d_temp_storage, flag_temp_storage_bytes);
//     // Run exclusive prefix sum
//     // cub::DeviceScan::ExclusiveSum(flag_d_temp_storage,
//     flag_temp_storage_bytes, flagArrSizeGlobal, flagArrOffsetGlobal,
//     numOfBlocks + 1);
//     // // Determine temporary device storage requirements
//     // void  *data_d_temp_storage     = NULL;
//     // size_t data_temp_storage_bytes = 0;
//     // cub::DeviceScan::ExclusiveSum(data_d_temp_storage,
//     data_temp_storage_bytes, compressedDataSizeGlobal,
//     compressedDataOffsetGlobal, numOfBlocks + 1);
//     // // Allocate temporary storage
//     // cudaMalloc(&data_d_temp_storage, data_temp_storage_bytes);
//     // // Run exclusive prefix sum
//     // cub::DeviceScan::ExclusiveSum(data_d_temp_storage,
//     data_temp_storage_bytes, compressedDataSizeGlobal,
//     compressedDataOffsetGlobal, numOfBlocks + 1);

//     // compressKernelIII<INPUT_TYPE><<<gridDim, blockDim, 0,
//     stream>>>(numOfBlocks, flagArrOffsetGlobal, compressedDataOffsetGlobal,
//     tmpFlagArrGlobal, tmpCompressedDataGlobal, flagArrGlobal,
//     compressedDataGlobal);

//     // decompressKernel<INPUT_TYPE><<<deGridDim, deBlockDim, 0,
//     stream>>>(output.data_ptr<INPUT_TYPE>(), numOfBlocks,
//     flagArrOffsetGlobal, compressedDataOffsetGlobal, flagArrGlobal,
//     compressedDataGlobal);
//     // // printf("%d\n", compressedDataSizeGlobal[0]);
//     // // cudaMemcpy((void *)output.data_ptr(), compressedDataOffsetGlobal,
//     sizeof(uint32_t) * (numOfBlocks + 1), cudaMemcpyDeviceToDevice);

//     // cudaFree(flagArrSizeGlobal);
//     // cudaFree(flagArrOffsetGlobal);
//     // cudaFree(compressedDataSizeGlobal);
//     // cudaFree(compressedDataOffsetGlobal);
//     // cudaFree(tmpFlagArrGlobal);
//     // cudaFree(tmpCompressedDataGlobal);
//     // cudaFree(flagArrGlobal);
//     // cudaFree(compressedDataGlobal);

//     return output;
// }

std::vector<torch::Tensor> lz_compress(torch::Tensor input,
                                       cudaStream_t stream) {
  uint32_t fileSize = input.nbytes();
  // calculate the padding size, unit in bytes
  uint32_t paddingSize = (BLOCK_SIZE - fileSize % BLOCK_SIZE) % BLOCK_SIZE;
  // calculate the datatype size, unit in datatype
  uint32_t datatypeSize = (fileSize + paddingSize) / sizeof(INPUT_TYPE);
  uint32_t numOfBlocks = (fileSize + paddingSize) / BLOCK_SIZE;
  torch::Tensor flagArrSizeGlobal = at::zeros(
      numOfBlocks + 1,
      torch::TensorOptions().device(input.device()).dtype(torch::kInt32));
  torch::Tensor flagArrOffsetGlobal = at::zeros(
      numOfBlocks + 1,
      torch::TensorOptions().device(input.device()).dtype(torch::kInt32));
  torch::Tensor compressedDataSizeGlobal = at::zeros(
      numOfBlocks + 1,
      torch::TensorOptions().device(input.device()).dtype(torch::kInt32));
  torch::Tensor compressedDataOffsetGlobal = at::zeros(
      numOfBlocks + 1,
      torch::TensorOptions().device(input.device()).dtype(torch::kInt32));
  torch::Tensor tmpFlagArrGlobal = at::zeros(
      sizeof(uint8_t) * datatypeSize / 8,
      torch::TensorOptions().device(input.device()).dtype(torch::kUInt8));
  torch::Tensor tmpCompressedDataGlobal = at::zeros(
      fileSize + paddingSize,
      torch::TensorOptions().device(input.device()).dtype(torch::kUInt8));
  torch::Tensor flagArrGlobal = at::zeros(
      sizeof(uint8_t) * datatypeSize / 8,
      torch::TensorOptions().device(input.device()).dtype(torch::kUInt8));
  torch::Tensor compressedDataGlobal = at::zeros(
      fileSize + paddingSize,
      torch::TensorOptions().device(input.device()).dtype(torch::kUInt8));
  torch::Tensor output = at::zeros(
      fileSize + paddingSize,
      torch::TensorOptions().device(input.device()).dtype(input.dtype()));
  // std::cout << (fileSize + paddingSize) << std::endl;
  dim3 gridDim(numOfBlocks);
  dim3 blockDim(THREAD_SIZE);

  int minEncodeLength = sizeof(INPUT_TYPE) == 1 ? 2 : 1;

  // launch kernels
  compressKernelI<INPUT_TYPE><<<gridDim, blockDim, 0, stream>>>(
      input.data_ptr<INPUT_TYPE>(), numOfBlocks,
      flagArrSizeGlobal.data_ptr<int>(),
      compressedDataSizeGlobal.data_ptr<int>(),
      tmpFlagArrGlobal.data_ptr<uint8_t>(),
      tmpCompressedDataGlobal.data_ptr<uint8_t>(), minEncodeLength);

  // Determine temporary device storage requirements
  void *flag_d_temp_storage = NULL;
  size_t flag_temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(flag_d_temp_storage, flag_temp_storage_bytes,
                                flagArrSizeGlobal.data_ptr<int>(),
                                flagArrOffsetGlobal.data_ptr<int>(),
                                numOfBlocks + 1, stream);
  // Allocate temporary storage
  cudaMalloc(&flag_d_temp_storage, flag_temp_storage_bytes);
  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(flag_d_temp_storage, flag_temp_storage_bytes,
                                flagArrSizeGlobal.data_ptr<int>(),
                                flagArrOffsetGlobal.data_ptr<int>(),
                                numOfBlocks + 1, stream);
  // Determine temporary device storage requirements
  void *data_d_temp_storage = NULL;
  size_t data_temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(data_d_temp_storage, data_temp_storage_bytes,
                                compressedDataSizeGlobal.data_ptr<int>(),
                                compressedDataOffsetGlobal.data_ptr<int>(),
                                numOfBlocks + 1, stream);
  // Allocate temporary storage
  cudaMalloc(&data_d_temp_storage, data_temp_storage_bytes);
  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(data_d_temp_storage, data_temp_storage_bytes,
                                compressedDataSizeGlobal.data_ptr<int>(),
                                compressedDataOffsetGlobal.data_ptr<int>(),
                                numOfBlocks + 1, stream);

  compressKernelIII<INPUT_TYPE><<<gridDim, blockDim, 0, stream>>>(
      numOfBlocks, flagArrOffsetGlobal.data_ptr<int>(),
      compressedDataOffsetGlobal.data_ptr<int>(),
      tmpFlagArrGlobal.data_ptr<uint8_t>(),
      tmpCompressedDataGlobal.data_ptr<uint8_t>(),
      flagArrGlobal.data_ptr<uint8_t>(),
      compressedDataGlobal.data_ptr<uint8_t>());

  // dim3 deGridDim((numOfBlocks + 31) / 32);
  // dim3 deBlockDim(32);
  // decompressKernel<INPUT_TYPE><<<deGridDim, deBlockDim, 0,
  // stream>>>(output.data_ptr<INPUT_TYPE>(),
  //                                                                    numOfBlocks,
  //                                                                    flagArrOffsetGlobal.data_ptr<int>(),
  //                                                                    compressedDataOffsetGlobal.data_ptr<int>(),
  //                                                                    flagArrGlobal.data_ptr<uint8_t>(),
  //                                                                    compressedDataGlobal.data_ptr<uint8_t>());

  std::vector<torch::Tensor> ret;
  // ret.push_back(output);
  ret.push_back(flagArrOffsetGlobal);
  ret.push_back(compressedDataOffsetGlobal);
  ret.push_back(flagArrGlobal);
  ret.push_back(compressedDataGlobal);
  return ret;
}

void lz_decompress(torch::Tensor output, int numOfBlocks,
                   torch::Tensor flagArrOffsetGlobal,
                   torch::Tensor compressedDataOffsetGlobal,
                   torch::Tensor flagArrGlobal,
                   torch::Tensor compressedDataGlobal, cudaStream_t stream) {
  dim3 deGridDim((numOfBlocks + 31) / 32);
  dim3 deBlockDim(32);
  decompressKernel<INPUT_TYPE><<<deGridDim, deBlockDim, 0, stream>>>(
      output.data_ptr<INPUT_TYPE>(), numOfBlocks,
      flagArrOffsetGlobal.data_ptr<int>(),
      compressedDataOffsetGlobal.data_ptr<int>(),
      flagArrGlobal.data_ptr<uint8_t>(),
      compressedDataGlobal.data_ptr<uint8_t>());
}