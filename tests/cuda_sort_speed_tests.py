import torch
from easyvolcap.utils.test_utils import my_tests
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer

# def test_cuda_sort_speed_incremental(N=1000000, repeat=10000):  # 1 mil
#     pts = torch.rand(int(N), device='cuda')

#     for i in tqdm(range(repeat)):
#         pts = pts.sort()[0]  # 6000 fps on 4090


# def test_cuda_argsort_speed(N=1000000, repeat=10000):  # 1 mil
#     pts = torch.rand(int(N), device='cuda')

#     for i in tqdm(range(repeat)):
#         idx = pts.argsort()  # 6000 fps on 4090

timer.disabled = False


def test_cuda_sort_speed(N=1000000, repeat=10000):  # 1 mil
    pts = torch.rand(int(N), device='cuda')

    timer.record()
    for i in tqdm(range(repeat)):
        ordered = pts.sort()[0]  # 6000 fps on 4090
    diff = timer.record()
    log(f'[CUDA] {N*repeat / diff / 1e6} MPPS')


def test_cpu_sort_speed(N=1000000, repeat=10):  # 1 mil
    pts = torch.rand(int(N), device='cpu')

    timer.record()
    for i in tqdm(range(repeat)):
        ordered = pts.sort()[0]  # 6000 fps on 4090
    diff = timer.record()
    log(f'[CPU] {N*repeat / diff / 1e6} MPPS')


def test_cpu_copy_sort_speed(N=1000000, repeat=10):  # 1 mil
    pts = torch.rand(int(N), device='cuda')

    timer.record()
    for i in tqdm(range(repeat)):
        pts_cpu = pts.cpu()
        ordered = pts_cpu.sort()[0]  # 6000 fps on 4090
        ordered = ordered.cuda()
    diff = timer.record()
    log(f'[CPU+Copy] {N*repeat / diff / 1e6} MPPS')


if __name__ == '__main__':
    my_tests(globals())
