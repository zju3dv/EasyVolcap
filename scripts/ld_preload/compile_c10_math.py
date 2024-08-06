import os
import torch
from torch.utils.cpp_extension import load

c10_math = load(
    name='c10_math',
    sources=[f'{os.path.dirname(__file__)}/../../easyvolcap/utils/src/c10_math.cpp'],
    extra_include_paths=[os.environ.get('CUDA_HOME', '/usr/local/cuda') + "/include"],
    extra_ldflags=['-ltorch_cuda'],
    extra_cuda_cflags=["--expt-relaxed-constexpr",
                       "-O2"],
    verbose=True
)
