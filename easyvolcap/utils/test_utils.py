from easyvolcap.utils.console_utils import *
from easyvolcap.utils.net_utils import setup_deterministic, take_gradient

import torch
import numpy as np
from typing import Callable


@catch_throw
def my_tests(globals: dict = globals(), prefix: str = 'test', fix_random: bool = False):
    # Setup deterministic testing environment
    setup_deterministic(fix_random)

    # Extract testing functions
    tests = {name: func for name, func in globals.items() if name.startswith(prefix)}

    # Run tests
    pbar = tqdm(total=len(tests))
    for name, func in tests.items():
        pbar.desc = name
        pbar.refresh()

        func()
        log(f'{name}: {green("OK")}')

        pbar.update(n=1)
        pbar.refresh()


def assert_true(expr):
    if isinstance(expr, torch.Tensor):
        expr = expr.all()
    assert expr, f'{repr(expr)} is not true'


def assert_func(func, *args, **kwargs):
    return func(*args, **kwargs)


def assert_array_equal(a: torch.Tensor, b: torch.Tensor):
    if isinstance(a, torch.Tensor): a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor): b = b.detach().cpu().numpy()
    return assert_func(np.testing.assert_array_equal, a, b)


def assert_array_compare(x: Callable, a: torch.Tensor, b: torch.Tensor):
    if isinstance(a, torch.Tensor): a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor): b = b.detach().cpu().numpy()
    return assert_func(np.testing.assert_array_compare, x, a, b)


def assert_allclose(a: torch.Tensor, b: torch.Tensor, **kwargs):
    return assert_func(torch.testing.assert_close, a, b, **kwargs)


def grad_check(inputs: List[torch.Tensor], torch_outputs: List[torch.Tensor], cuda_outputs: List[torch.Tensor], **kwargs):
    torch_grads = []
    for input in inputs:
        for output in torch_outputs:
            torch_grads.append(take_gradient(output, input, create_graph=False))  # only once

    cuda_grads = []
    for input in inputs:
        for output in cuda_outputs:
            cuda_grads.append(take_gradient(output, input, create_graph=False))  # only once

    for i, (torch_grad, cuda_grad) in enumerate(zip(torch_grads, cuda_grads)):
        assert_allclose(torch_grad, cuda_grad, **kwargs)
