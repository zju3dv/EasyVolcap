import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.utils.test_utils import my_tests, assert_allclose, grad_check
from easyvolcap.utils.console_utils import *


from easyvolcap.utils.fdgs_utils import FDGSModel
from easyvolcap.utils.sh_utils import eval_shfs_4d, sh_channels, sh_channels_4d
from easyvolcap.utils.gaussian_utils import prepare_gaussian_camera, convert_to_gaussian_camera, prepare_cpu_gaussian_camera, convert_to_cpu_gaussian_camera, render_diff_gauss, build_cov6, render_fast_gauss
from easyvolcap.utils.tfgs_utils import compute_cov_4d, compute_tile_avg
from easyvolcap.utils.net_utils import take_gradient, make_params, setup_deterministic
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.data_utils import save_image
import diff_gauss


setup_deterministic()


N = 1000000
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
scale = torch.rand(N, 3, device='cuda') * 0.25
opacity = torch.ones(N, 1, device='cuda') * 0.95
rotation = torch.rand(N, 4, device='cuda')
norm = xyz3.norm(dim=-1, keepdim=True)
denom = ((norm / norm.max()) * 400).int()
cov6 = build_cov6(scale, rotation)


def test_tile_mask_speed():
    camera = convert_to_gaussian_camera(K, R, T, H, W, n, f, K.cpu(), R.cpu(), T.cpu(), H.cpu(), W.cpu(), n.cpu(), f.cpu())
    REPEAT = 100
    xyz3.requires_grad_()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    rgb, _, _, _ = render_diff_gauss(xyz3, rgb3, cov6, opacity, camera)
    for i in tqdm(range(REPEAT), desc='full backward'):
        grad = take_gradient(rgb, xyz3)
        torch.cuda.synchronize()
    log('full backward peak memory:', torch.cuda.max_memory_allocated() // 2**20)

    for i in tqdm(range(REPEAT), desc='full forward'):
        rgb, _, _, _ = render_diff_gauss(xyz3, rgb3, cov6, opacity, camera)
    save_image('full_rgb.png', rgb)


if __name__ == '__main__':
    my_tests(globals(), fix_random=True)
