import gc
import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.utils.test_utils import my_tests, assert_allclose, grad_check
from easyvolcap.utils.console_utils import *


def test_pinning_parameters():
    N_LEVELS = 1
    N_NODES = 100
    N_PARAMS = 9
    model = nn.Module()
    sampler = nn.Module()
    network = nn.Module()
    renderer = nn.Module()
    tfgs = nn.Module()
    tfgs.forest = nn.ParameterList([
        nn.ParameterList([
            nn.ParameterDict({
                str(k):
                    torch.empty(0, 4, 2**19)
                for k in range(N_PARAMS)
            }) for j in range(N_NODES)
        ]) for i in tqdm(range(N_LEVELS))
    ])

    network.tfgs = tfgs
    sampler.unregistered = [network]
    renderer.unregistered = [network]
    model.sampler = sampler
    model.network = network
    model.renderer = renderer

    data = torch.rand(1, 4, 2**19, N_LEVELS, N_NODES, N_PARAMS)

    length = len(list(model.named_parameters()))
    pbar = tqdm(total=length)
    # # with torch.no_grad():
    #     for name, param in model.named_parameters():
    #         # param.data = torch.cat([param.data, torch.rand((1,) + param.data.shape[1:])], dim=0)
    #         # param.data = torch.rand((1,) + param.data.shape[1:])
    #         param.set_(torch.rand((1,) + param.data.shape[1:]))
    #         # param = nn.Parameter(torch.rand((1,) + param.data.shape[1:]), requires_grad=param.requires_grad)
    with torch.no_grad():
        for l in range(N_LEVELS):
            for n in range(N_NODES):
                for p in range(N_PARAMS):
                    # param = model.tfgs[l][n][str(p)]
                    param = getattr(model.network.tfgs.forest[l][n], str(p))
                    # param.data = data[:, :, :, l, n, p]
                    # param.data = data[:, :, torch.randperm(2**19)].contiguous()
                    # param.data = torch.rand((1,) + param.data.shape[1:])
                    # param.set_(torch.rand((1,) + param.data.shape[1:]))
                    # param.data = torch.cat([param.data, torch.rand((1,) + param.data.shape[1:])], dim=0)
                    param.data = torch.cat([param.data, data[:, :, :, l, n, p].contiguous()], dim=0)
                    pbar.update()
    pbar.close()

    del data
    gc.collect()
    # torch.cuda.empty_cache()
    # from easyvolcap.utils.host_utils import host_empty_cache
    # host_empty_cache()

    # model.to('cpu', non_blocking=True)

    pbar = tqdm(total=length)
    for name, param in model.named_parameters():
        param.data = param.data.to('cuda', non_blocking=True).to('cpu', non_blocking=True)
        # p.data = p.data.pin_memory()
        pbar.update()
    pbar.close()
    breakpoint()


def test_minimal_pinning():
    N_LEVELS = 9
    N_NODES = 100
    N_PARAMS = 9
    data = torch.rand(1, 2**16, 4, N_LEVELS, N_NODES, N_PARAMS)

    module = nn.Module()
    module.copied = torch.empty(0, 2**16, 4, N_LEVELS, N_NODES, N_PARAMS)
    module.copied.data = torch.cat([module.copied.data, data], dim=0)

    del data
    gc.collect()

    module.copied.data = module.copied.data.pin_memory()

    breakpoint()


def test_pin_memory_release_behavior():
    REPEAT = 128
    N = 1024
    abc = dotdict()
    data_stream = torch.cuda.Stream()
    from easyvolcap.utils.host_utils import host_empty_cache
    for i in tqdm(range(REPEAT)):
        with torch.cuda.stream(data_stream):
            gc.collect()
            # torch.xpu.empty_cache()
            torch.cuda.empty_cache()
            host_empty_cache()
            abc.edf = torch.empty(N, N, N + i, device='cpu', pin_memory=True)
        # abc.edf.copy_(torch.rand(N, N, N + i, device='cuda'), non_blocking=True)
        # abc.edf = torch.rand(N, N, N + i, device='cuda').to('cpu', non_blocking=True)  # 4GB
        # torch.cuda.synchronize()
        # gc.collect()
    # breakpoint()


if __name__ == '__main__':
    my_tests(globals())
