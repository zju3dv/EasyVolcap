// Exposes the host allocator API
#include <torch/extension.h>
#include <ATen/cuda/CachingHostAllocator.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("host_empty_cache", &at::cuda::CachingHostAllocator_emptyCache, "Empty the cache of caching host allocator");
}