#ifndef JIT
#define JIT

#include <torch/extension.h>
#if defined(USE_GPU)
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <regex>

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
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

namespace jit {

extern int mem_stride_copy_char_fd;
extern int mem_stride_copy_uint4_fd;
extern int mem_stride_copy_gridsize;
extern int mem_stride_copy_blocksize;

inline static std::string file_read(const char *path) {
    FILE *fp = fopen(path, "rb");
    CHECK_EQ(true, fp != nullptr);
    fseek(fp, 0, SEEK_END);
    size_t code_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::string code;
    code.resize(code_size);
    CHECK_EQ(code_size, fread((void *)code.data(), 1, code_size, fp));
    fclose(fp);
    return code;
}

inline static void file_write(const char *path, const std::string &code) {
    FILE *fp = fopen(path, "wb");
    CHECK_EQ(true, fp != nullptr);
    CHECK_EQ(code.size(), fwrite((void *)code.data(), 1, code.size(), fp));
    fclose(fp);
}

extern std::string __sdk_home__;

static void update_sdk_home(const torch::Tensor &sdk_path) {
    CHECK_CPU(sdk_path);
    __sdk_home__ = static_cast<char *>(sdk_path.data_ptr());
}

inline std::string sdk_path(const std::string &rel = "") {
    static std::string cuda_home, cc;
    if (cuda_home.size() == 0) {
#if !defined(__HIP_PLATFORM_HCC__)
        cc = "bin/nvcc";
#else
        cc        = "bin/hipcc";
#endif

#if defined(__linux__)
        cuda_home = __sdk_home__ + std::string("/");
#else
        cuda_home = __sdk_home__ + std::string("\\");
#endif
    }
    if (rel.size() > 0)
        return cuda_home + rel;
    return cuda_home + cc;
}

static std::string nvcc_compile(const char *code, const std::string &arch) {
#if defined(__linux__)
    char code_path[] = "/tmp/torch-tutel-XXXXXX.cu";
    CHECK_NE(-1, mkstemps(code_path, 3));

    file_write(code_path, code);
    std::string fatbin_path = code_path + std::string(".fatbin");

    std::string entry = sdk_path();
    if (access(entry.c_str(), F_OK) != 0) {
        LOG(FATAL) << "Failed to detect CUDA compiler file: " << entry << ", please set CUDA_HOME environment to configure CUDA SDK location correctly.";
        exit(1);
    }
    pid_t pid = fork();
    if (pid == 0) {
#if !defined(__HIP_PLATFORM_HCC__)
        CHECK_EQ(-1, execl(entry.c_str(), entry.c_str(), code_path, "-o", fatbin_path.c_str(), "--fatbin", "-O4", "-gencode", ("arch=compute_" + arch + ",code=sm_" + arch).c_str(), (char *)NULL));
#else
        CHECK_EQ(-1, execl(entry.c_str(), entry.c_str(), code_path, "-o", fatbin_path.c_str(), "--genco", "-O4", "-w", ("--amdgpu-target=" + arch).c_str(), (char *)NULL));
#endif
        exit(1);
    } else {
        wait(NULL);
    }
    auto image = file_read(fatbin_path.data());
    unlink(fatbin_path.data());
    unlink(code_path);
    return image;
#else
    return "";
#endif
}

static std::string nvrtc_compile(const char *code, const std::string &arch) {
#if !defined(__HIP_PLATFORM_HCC__)
    std::string               arch_option = "--gpu-architecture=compute_" + arch, include_path = "--include-path=" + sdk_path("include");
    std::vector<const char *> param_cstrings = {"--restrict", include_path.c_str(), arch_option.c_str(), "--use_fast_math", "--extra-device-vectorization"};
#else
    std::string arch_option = "--gpu-architecture=" + arch;
    std::vector<const char *> param_cstrings = {arch_option.c_str(), "-O4"};
#endif
    nvrtcProgram prog;

    CHECK_EQ(0, nvrtcCreateProgram(&prog, code, nullptr, 0, nullptr, nullptr));
    nvrtcResult res = nvrtcCompileProgram(prog, param_cstrings.size(), param_cstrings.data());

    size_t log_size;
    CHECK_EQ(0, nvrtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size);
    CHECK_EQ(0, nvrtcGetProgramLog(prog, &log[0]));
    if (0 != res) {
        static bool once_flag = false;
        if (!once_flag) {
            once_flag = true;
            LOG(WARNING) << log << " Failed to use NVRTC for JIT compilation in this Pytorch version, try another approach using CUDA compiler.. (To always disable NVRTC, please: export USE_NVRTC=0)";
        }
        CHECK_EQ(0, nvrtcDestroyProgram(&prog));
        return "";
    }

    size_t ptx_size;
    CHECK_EQ(0, nvrtcGetPTXSize(prog, &ptx_size));

    std::string ptx;
    ptx.resize(ptx_size);
    CHECK_EQ(0, nvrtcGetPTX(prog, &ptx[0]));
    CHECK_EQ(0, nvrtcDestroyProgram(&prog));
    return ptx;
}

struct ModuleConfig {
    // Handling JIT compilation in Multi-gpu cases
    std::vector<CUfunction> hFunc;
    std::string             code, fname;
    dim3                    blocks, threads;
};

extern std::vector<ModuleConfig> _gms;

inline static CUfunction jit_activate(int fd, int dev) {
//std::cout << (&_gms) << " " << fd << std::endl;
    auto &gm = _gms[fd];
    if (gm.hFunc.size() <= dev)
        gm.hFunc.resize(dev + 1);

    if (gm.hFunc[dev] == nullptr) {
#if !defined(__HIP_PLATFORM_HCC__)
        int major, minor;
        CHECK_EQ(0, cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
        CHECK_EQ(0, cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
        std::string arch = std::to_string(major) + std::to_string(minor);
#else
        hipDeviceProp_t prop;
        CHECK_EQ(0, hipGetDeviceProperties(&prop, dev));
        std::string arch = prop.gcnArchName;
#endif
        const char *source = gm.code.data(), *pos, *tail;

        int         use_nvrtc = getenv("USE_NVRTC") ? std::atoi(getenv("USE_NVRTC")) : 0;
        std::string image;
        if (use_nvrtc || (image = nvcc_compile(source, arch)) == "") {
            image = nvrtc_compile(source, arch);
        }

        long launch_bound;
        {
            char        tag[] = " __launch_bounds__(";
            const char *pos   = strstr(source, tag);
            launch_bound      = pos ? std::atol(pos + sizeof(tag) - 1) : 1024L;
        }

        static CUjit_option options[] = {CU_JIT_OPTIMIZATION_LEVEL, CU_JIT_THREADS_PER_BLOCK};
        static void        *values[]  = {(void *)4L, (void *)launch_bound};

        CUmodule hMod = nullptr;
        CHECK_EQ(0, cuModuleLoadDataEx(&hMod, image.c_str(), sizeof(options) / sizeof(*options), options, values));
        CHECK_NE(nullptr, hMod);

        CHECK_NE(nullptr, (pos = strstr(source, " void ")));
        pos += 6;
        CHECK_NE(nullptr, (tail = strchr(pos, '(')));

        std::string fname = std::string(pos, tail - pos);
        gm.fname          = fname;
        CHECK_EQ(0, cuModuleGetFunction(&gm.hFunc[dev], hMod, fname.c_str()));
        CHECK_NE(nullptr, gm.hFunc[dev]);
    }

    return gm.hFunc[dev];
}

static void jit_execute(const std::vector<const void *> &ppargs, int fd, int dev, const dim3 &blocks, const dim3 &threads, cudaStream_t stream = 0) {
    CUfunction hfunc = jit_activate(fd, dev);
    CHECK_EQ(0, cuLaunchKernel(hfunc, blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, 0, stream, (void **)ppargs.data(), nullptr));
}

static void jit_execute_with_values(const std::vector<const void *> &pargs, int fd, int dev, const dim3 &blocks, const dim3 &threads, cudaStream_t stream = 0) {
    std::vector<const void *> ppargs(pargs.size());
    for (int i = 0; i < ppargs.size(); ++i)
        ppargs[i] = &pargs[i];
    jit_execute(ppargs, fd, dev, blocks, threads, stream);
}

static int inject_source(const std::string &headless_code) {
    int fd = _gms.size();
    _gms.resize(fd + 1);

    auto &gm = _gms[fd];
#if !defined(__HIP_PLATFORM_HCC__)
    gm.code = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n" + headless_code;
#else
    gm.code = "#include <hip/hip_runtime.h>\n" + headless_code;
#endif

    const char *source = headless_code.c_str();
    {
        char        tag[] = "// [thread_extent] blockIdx.x = ";
        const char *pos   = strstr(source, tag);
        gm.blocks.x       = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
    }
    {
        char        tag[] = "// [thread_extent] blockIdx.y = ";
        const char *pos   = strstr(source, tag);
        gm.blocks.y       = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
    }
    {
        char        tag[] = "// [thread_extent] blockIdx.z = ";
        const char *pos   = strstr(source, tag);
        gm.blocks.z       = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
    }
    {
        char        tag[] = "// [thread_extent] threadIdx.x = ";
        const char *pos   = strstr(source, tag);
        gm.threads.x      = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
    }
    {
        char        tag[] = "// [thread_extent] threadIdx.y = ";
        const char *pos   = strstr(source, tag);
        gm.threads.y      = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
    }
    {
        char        tag[] = "// [thread_extent] threadIdx.z = ";
        const char *pos   = strstr(source, tag);
        gm.threads.z      = pos ? std::atoi(pos + sizeof(tag) - 1) : 1;
    }

    return fd;
}

static void invoke(const std::vector<torch::Tensor> &ts, const std::vector<long> &args, const std::vector<int> &blocks, int fd) {
    std::vector<const void *> pargs(ts.size() + args.size()), ppargs(ts.size() + args.size());
    for (int i = 0; i < (int)ts.size(); ++i) {
        CHECK_CUDA(ts[i]);
        pargs[i] = ts[i].data_ptr(), ppargs[i] = &pargs[i];
    }
    for (int i = (int)ts.size(); i < (int)pargs.size(); ++i) {
        pargs[i] = (void *)args[i - ts.size()], ppargs[i] = &pargs[i];
    }

    int dev = ts[0].device().index();
    CHECK_EQ(0, cudaSetDevice(dev));
    if (blocks.size() == 0)
        jit_execute(ppargs, fd, dev, _gms[fd].blocks, _gms[fd].threads, at::cuda::getDefaultCUDAStream().stream());
    else if (blocks.size() == 1)
        jit_execute(ppargs, fd, dev, dim3(blocks[0]), _gms[fd].threads, at::cuda::getDefaultCUDAStream().stream());
    else if (blocks.size() == 2)
        jit_execute(ppargs, fd, dev, dim3(blocks[0], blocks[1]), _gms[fd].threads, at::cuda::getDefaultCUDAStream().stream());
    else
        jit_execute(ppargs, fd, dev, dim3(blocks[0], blocks[1], blocks[2]), _gms[fd].threads, at::cuda::getDefaultCUDAStream().stream());
}

inline void jit_init(int g_local_rank){
    if (mem_stride_copy_uint4_fd == -1) {
        std::string mem_stride_copy_cu = R"(
extern "C" __global__ void memStrideCopyKernel(
    $T *__restrict__ out, const $T *__restrict__ in,
    const size_t size, const int height, const int width) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < size * height * width; i += gridDim.x * blockDim.x) {
        const size_t index = i / size, offset = i % size;
        const size_t j = (width * (index % height) + (index / height)) * size + offset;
        out[j] = in[i];
    }
}
    )";
        mem_stride_copy_char_fd        = inject_source(std::regex_replace(mem_stride_copy_cu, std::regex("\\$T"), "char"));
        mem_stride_copy_uint4_fd       = inject_source(std::regex_replace(mem_stride_copy_cu, std::regex("\\$T"), "uint4"));
        CHECK_NE(-1, mem_stride_copy_char_fd);
        CHECK_NE(-1, mem_stride_copy_uint4_fd);
        CUfunction hfunc = jit_activate(mem_stride_copy_uint4_fd, g_local_rank);
#if !defined(__HIP_PLATFORM_HCC__)
        CHECK_EQ(0, cuOccupancyMaxPotentialBlockSize(&mem_stride_copy_gridsize, &mem_stride_copy_blocksize, hfunc, 0, 0, 0));
#else
        CHECK_EQ(0, hipModuleOccupancyMaxPotentialBlockSize(&mem_stride_copy_gridsize, &mem_stride_copy_blocksize, hfunc, 0, 0));
#endif
    }
}


} // namespace jit
#endif

#endif