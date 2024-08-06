// Exposes the host allocator API
#include <torch/extension.h>
#include <c10/util/llvmMathExtras.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("power_of_2_ceil", &c10::llvm::PowerOf2Ceil, "PowerOf2Ceil");
}