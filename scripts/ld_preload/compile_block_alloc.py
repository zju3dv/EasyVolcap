import os
import torch
from torch.utils.cpp_extension import load

block_alloc = load(
    name='block_alloc',
    sources=[f'{os.path.dirname(__file__)}/../../easyvolcap/utils/src/block_alloc.cpp'],
    extra_include_paths=[os.environ.get('CUDA_HOME', '/usr/local/cuda') + "/include"],
    extra_ldflags=['-ltorch_cuda'],
    extra_cuda_cflags=["--expt-relaxed-constexpr",
                       "-O2"],
    verbose=True
)
