#include <cstdint>
#include <iostream>

// c++ -fPIC -c PowerOf2Ceil.cpp -O2 -o PowerOf2Ceil.so

// Exposes the host allocator API
constexpr uint64_t BLOCK_SIZE = 512;

namespace c10::llvm {

  /// Returns the power of two which is greater than or equal to the given value.
  /// Essentially, it is a ceil operation across the domain of powers of two.
  uint64_t PowerOf2Ceil(uint64_t A) {
    std::cout << "Input size: " << A << std::endl;
    return (A + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  }

}
