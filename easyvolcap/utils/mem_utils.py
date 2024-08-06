import gc
import math
import torch
from torch import nn
# from typing import List
# from easyvolcap.utils.console_utils import *

LEN = 79

# some pytorch low-level memory management constant
# the minimal allocate memory size (Byte)
PYTORCH_MIN_ALLOCATE = 2 ** 9
# the minimal cache memory size (Byte)
PYTORCH_MIN_CACHE = 2 ** 20


def calc_memory_usage(tensor: torch.Tensor):
    numel = tensor.numel()
    element_size = tensor.element_size()
    fact_numel = tensor.storage().size()
    fact_memory_size = fact_numel * element_size
    # since pytorch allocate at least 512 Bytes for any tensor, round
    # up to a multiple of 512
    memory_size = math.ceil(fact_memory_size / PYTORCH_MIN_ALLOCATE) \
        * PYTORCH_MIN_ALLOCATE
    return memory_size


def check_pinned_memory_usage():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if isinstance(obj, torch.Tensor) and obj.is_pinned()]
    return sum(calc_memory_usage(tensor) for tensor in tensors)


def check_device_memory_usage(device: torch.device = torch.device('cuda')):
    objects = gc.get_objects()
    tensors = [obj for obj in objects if isinstance(obj, torch.Tensor) and obj.device == device]
    return sum(calc_memory_usage(tensor) for tensor in tensors)


def check_cpu_memory_usage():
    return check_device_memory_usage(torch.device('cpu'))


def check_cuda_memory_usage():
    return check_device_memory_usage(torch.device('cuda'))
