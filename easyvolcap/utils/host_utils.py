import os
import torch
from torch.utils.cpp_extension import load

host_alloc = load(
    name='host_alloc',
    sources=[f'{os.path.dirname(__file__)}/src/host_alloc.cpp'],
    extra_include_paths=[os.environ.get('CUDA_HOME', '/usr/local/cuda') + "/include"],
    extra_ldflags=['-ltorch_cuda'],
    extra_cuda_cflags=["--expt-relaxed-constexpr",
                       "-O2"],
    verbose=True
)


def host_empty_cache():
    host_alloc.host_empty_cache()
