// This file intends to replace the PowerOfTwo caching behavior of the default CachingHostAllocator on CUDA for PyTorch

#include <torch/extension.h>
#include <ATen/cuda/CachingHostAllocator.h>


// Exposes the host allocator API
constexpr uint64_t BLOCK_SIZE = 512;

namespace at::cuda {

  /// Returns the power of two which is greater than or equal to the given value.
  /// Essentially, it is a ceil operation across the domain of powers of two.
  struct CUDAHostAllocatorWrapper {

    std::pair<void*, void*> allocate(size_t size) {
      // return (size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
      std::cout << "Input size: " << size << std::endl;

      if (size == 0) {
        return {nullptr, nullptr};
      }

      // process_events();

      // // First, try to allocate from the free list
      // auto* block = get_free_block(size);
      // if (block) {
      //   return {block->ptr_, reinterpret_cast<void*>(block)};
      // }

      // // Round up the allocation to the nearest power of two to improve reuse.
      // size_t roundSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
      // std::cout << "Round size: " << roundSize << std::endl;
      // void* ptr = nullptr;
      // allocate_host_memory(roundSize, &ptr);

      // // Then, create a new block.
      // block = new at::HostBlock(roundSize, ptr);
      // block->allocated_ = true;

      // add_allocated_block(block);
      // return {block->ptr_, reinterpret_cast<void*>(block)};
    }

  };

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("host_alloc", &at::cuda::HostAlloc, "Alloc this amount of memory");
}