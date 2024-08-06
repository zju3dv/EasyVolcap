import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.utils.test_utils import my_tests, assert_allclose, grad_check
from easyvolcap.utils.console_utils import *


from easyvolcap.utils.fdgs_utils import FDGSModel
from easyvolcap.utils.sh_utils import eval_shfs_4d, sh_channels, sh_channels_4d
from easyvolcap.utils.gaussian_utils import prepare_gaussian_camera, convert_to_gaussian_camera, prepare_cpu_gaussian_camera, convert_to_cpu_gaussian_camera, render_diff_gauss, build_cov6, render_fast_gauss
from easyvolcap.utils.tfgs_utils import compute_cov_4d, compute_tile_avg
from easyvolcap.utils.net_utils import take_gradient, make_params
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.data_utils import save_image
import diff_gauss

N = 10000  # be faster, be agile

xyz = torch.rand(N, 3, device='cuda') * 10 - 5
t = torch.rand(N, 1, device='cuda') * 10
rgb = torch.rand(N, 3, device='cuda') * 10 - 5
init_scale_t = torch.rand(N, 1, device='cuda') * 0.0166666667 ** 2

fdgs = FDGSModel(xyz, t, rgb, init_scale_t=init_scale_t).cuda()
fdgs._rotation_l = make_params(F.normalize(torch.rand_like(fdgs._rotation_l), dim=-1))
fdgs._rotation_r = make_params(F.normalize(torch.rand_like(fdgs._rotation_r), dim=-1))


def test_cuda_covariance():
    scaling_xyzt = fdgs.scaling_activation(torch.cat([fdgs._scaling, fdgs._scaling_t], dim=-1))
    rotation_l = fdgs._rotation_l * 3.0
    rotation_r = fdgs._rotation_r * 3.0
    torch_cov, torch_ms, torch_cov_t = compute_cov_4d(scaling_xyzt, rotation_l, rotation_r)
    cuda_cov, cuda_ms, cuda_cov_t = diff_gauss.compute_cov_4d(scaling_xyzt, rotation_l, rotation_r)

    assert_allclose(torch_cov, cuda_cov)
    assert_allclose(torch_ms, cuda_ms)
    assert_allclose(torch_cov_t, cuda_cov_t)


def test_cuda_covariance_backward():
    scaling_xyzt = fdgs.scaling_activation(torch.cat([fdgs._scaling, fdgs._scaling_t], dim=-1))
    rotation_l = fdgs._rotation_l * 3.0
    rotation_r = fdgs._rotation_r * 3.0
    torch_cov, torch_ms, torch_cov_t = compute_cov_4d(scaling_xyzt, rotation_l, rotation_r)
    cuda_cov, cuda_ms, cuda_cov_t = diff_gauss.compute_cov_4d(scaling_xyzt, rotation_l, rotation_r)

    inputs = [fdgs._scaling, fdgs._scaling_t, fdgs._rotation_l, fdgs._rotation_r]
    torch_outputs = [torch_cov, torch_ms, torch_cov_t]
    cuda_outputs = [cuda_cov, cuda_ms, cuda_cov_t]
    grad_check(inputs, torch_outputs, cuda_outputs, atol=1e-4, rtol=1.3e-6)


sh_deg = 3
sh_deg_t = 2
sh = (torch.rand(N, 3, (sh_deg + 1)**2 * (sh_deg_t + 1), device='cuda') * 2 - 1).requires_grad_()
dir = normalize(torch.rand(N, 3, device='cuda') * 2 - 1).requires_grad_()
dir_t = (torch.rand(N, 1, device='cuda') * 2 - 1).requires_grad_()
l = 1.0


def test_cuda_sh4d():
    torch_rgb = (eval_shfs_4d(sh_deg, sh_deg_t, sh, dir, dir_t, torch.as_tensor(l).to('cuda', non_blocking=True)) + 0.5)
    cuda_rgb = diff_gauss.compute_sh_4d(sh_deg, sh_deg_t, sh.mT, dir, dir_t, l)
    assert_allclose(torch_rgb, cuda_rgb)


def test_cuda_sh4d_backward():
    torch_rgb = (eval_shfs_4d(sh_deg, sh_deg_t, sh, dir, dir_t, torch.as_tensor(l).to('cuda', non_blocking=True)) + 0.5)
    cuda_rgb = diff_gauss.compute_sh_4d(sh_deg, sh_deg_t, sh.mT, dir, dir_t, l)

    inputs = [sh, dir, dir_t]
    torch_outputs = [torch_rgb]
    cuda_outputs = [cuda_rgb]
    grad_check(inputs, torch_outputs, cuda_outputs, atol=1e-4, rtol=1.3e-6)


N = 100000
BLOCK_SIZE_Y = 16
BLOCK_SIZE_X = 16

K = torch.as_tensor([[736.5288696289062, 0.0, 682.7473754882812], [0.0, 736.4380493164062, 511.99737548828125], [0.0, 0.0, 1.0]], dtype=torch.float, device='cuda')
R = torch.as_tensor([[0.9938720464706421, 0.0, -0.11053764075040817], [-0.0008741595083847642, 0.9999688267707825, -0.007859790697693825], [0.1105341762304306, 0.007908252067863941, 0.9938408732414246]], dtype=torch.float, device='cuda')
T = torch.as_tensor([[-0.2975313067436218], [-1.2581647634506226], [0.2818146347999573]], dtype=torch.float, device='cuda')
n = torch.as_tensor(2, dtype=torch.float, device='cuda')
f = torch.as_tensor(3, dtype=torch.float, device='cuda')
W = torch.as_tensor(1366, dtype=torch.long, device='cuda')
H = torch.as_tensor(768, dtype=torch.long, device='cuda')

xyz3 = torch.rand(N, 3, device='cuda') * 10 - 5
rgb3 = torch.rand(N, 3, device='cuda')
scale = torch.rand(N, 3, device='cuda') * 0.1
opacity = torch.ones(N, 1, device='cuda') * 0.1
rotation = torch.rand(N, 4, device='cuda')
norm = xyz3.norm(dim=-1, keepdim=True)
denom = ((norm / norm.max()) * 400).int()


def test_tile_mask_generation():
    Ht, Wt = (H + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y, (W + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    Ht, Wt = torch.as_tensor(Ht, dtype=torch.long, device='cuda'), torch.as_tensor(Wt, dtype=torch.long, device='cuda')
    Kt = K.clone()
    Kt[:2] /= 16
    camera = convert_to_gaussian_camera(Kt, R, T, Ht, Wt, n, f, Kt.cpu(), R.cpu(), T.cpu(), Ht.cpu(), Wt.cpu(), n.cpu(), f.cpu())

    REPEAT = 1000
    global cov6, tile_mask
    cov6 = build_cov6(scale, rotation)
    for i in tqdm(range(REPEAT)):
        tile_acc, _, _, _ = render_diff_gauss(xyz3, (denom / 400 < 0.25).expand(rgb3.shape), cov6, opacity, camera)
    tile_mask = tile_acc[..., :1] > 0.5
    log('tile_mask valid ratio:', tile_mask.sum().item() / tile_mask.numel())
    log('tile_mask:', tile_mask.sum().item(), '/', tile_mask.numel(), 'min:', tile_acc.min().item(), 'max:', tile_acc.max().item())
    save_image('tile_acc.png', tile_acc)
    save_image('tile_msk.png', tile_mask)


def test_tile_mask_speed():
    camera = convert_to_gaussian_camera(K, R, T, H, W, n, f, K.cpu(), R.cpu(), T.cpu(), H.cpu(), W.cpu(), n.cpu(), f.cpu())
    REPEAT = 1000
    xyz3.requires_grad_()
    mock_tile = torch.ones_like(tile_mask)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rgb, _, _, _ = render_diff_gauss(xyz3, rgb3, cov6, opacity, camera, tile_mask)
    for i in tqdm(range(REPEAT), desc='tile backward'):
        grad = take_gradient(rgb, xyz3)
    log('tile backward peak memory:', torch.cuda.max_memory_allocated() // 2**20)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rgb, _, _, _ = render_diff_gauss(xyz3, rgb3, cov6, opacity, camera, mock_tile)
    for i in tqdm(range(REPEAT), desc='mock backward'):
        grad = take_gradient(rgb, xyz3)
    log('mock backward peak memory:', torch.cuda.max_memory_allocated() // 2**20)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rgb, _, _, _ = render_diff_gauss(xyz3, rgb3, cov6, opacity, camera)
    for i in tqdm(range(REPEAT), desc='full backward'):
        grad = take_gradient(rgb, xyz3)
    log('full backward peak memory:', torch.cuda.max_memory_allocated() // 2**20)

    for i in tqdm(range(REPEAT), desc='tile forward'):
        rgb, _, _, _ = render_diff_gauss(xyz3, rgb3, cov6, opacity, camera, tile_mask)
    save_image('tile_rgb.png', rgb)

    for i in tqdm(range(REPEAT), desc='mock forward'):
        rgb, _, _, _ = render_diff_gauss(xyz3, rgb3, cov6, opacity, camera, mock_tile)
    save_image('mock_rgb.png', rgb)

    for i in tqdm(range(REPEAT), desc='full forward'):
        rgb, _, _, _ = render_diff_gauss(xyz3, rgb3, cov6, opacity, camera)
    save_image('full_rgb.png', rgb)


REPEAT = 1000


def test_fast_gauss_speed():
    Kt = K.clone()
    Kt[:2] *= 4
    Ht, Wt = H * 4, W * 4
    camera = convert_to_gaussian_camera(Kt, R, T, Ht, Wt, n, f, Kt.cpu(), R.cpu(), T.cpu(), Ht.cpu(), Wt.cpu(), n.cpu(), f.cpu())
    cpu_camera = convert_to_cpu_gaussian_camera(Kt, R, T, Ht, Wt, n, f, Kt.cpu(), R.cpu(), T.cpu(), Ht.cpu(), Wt.cpu(), n.cpu(), f.cpu())
    cov6 = build_cov6(scale, rotation)

    xyz3.requires_grad_(False)

    rgb, _, _, _ = render_diff_gauss(xyz3, rgb3, cov6, opacity, camera)
    for i in tqdm(range(REPEAT), desc='diff_gauss'):
        rgb, _, _, _ = render_diff_gauss(xyz3, rgb3, cov6, opacity, camera)

    rgb, _, _, _ = render_fast_gauss(xyz3, rgb3, cov6, opacity, cpu_camera)
    from fast_gauss import raster_context
    raster_context.offline_writeback = False
    for i in tqdm(range(REPEAT), desc='fast_gauss'):
        rgb, _, _, _ = render_fast_gauss(xyz3, rgb3, cov6, opacity, cpu_camera)


if __name__ == '__main__':
    my_tests(globals(), fix_random=True)
