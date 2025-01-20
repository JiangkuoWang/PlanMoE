#include "jit.h"

namespace jit {

int mem_stride_copy_char_fd = -1;
int mem_stride_copy_uint4_fd = -1;
int mem_stride_copy_gridsize = 1;
int mem_stride_copy_blocksize = 1;
std::string __sdk_home__;
std::vector<ModuleConfig> _gms;

}