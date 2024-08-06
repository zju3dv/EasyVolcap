import torch
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.test_utils import my_tests
from easyvolcap.utils.adam_utils import fused_adam, adam


def test_adam_custom_vs_pytorch():
    # Tensor sizes and parameters
    size = 1024
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    # Initialize tensors for custom kernel
    param_custom = torch.randn(size, device='cuda')
    grad = torch.randn(size, device='cuda')
    exp_avg = torch.rand_like(param_custom)
    exp_avg_sq = torch.rand_like(param_custom)
    step_t = torch.as_tensor(0, dtype=torch.float)

    # Initialize tensors for PyTorch Adam
    param_pytorch = param_custom.clone().detach()
    exp_avg_pt = exp_avg.clone().detach()
    exp_avg_sq_pt = exp_avg_sq.clone().detach()
    step_t_pt = step_t.clone().detach()

    # Apply custom CUDA Adam update
    fused_adam(
        param_custom, grad, exp_avg, exp_avg_sq, step_t,
        beta1, beta2, lr, eps
    )

    adam(
        param_pytorch, grad, exp_avg_pt, exp_avg_sq_pt, step_t_pt,
        beta1, beta2, lr, eps
    )

    # Compare results
    print("Custom CUDA Adam vs PyTorch Adam")
    print("Parameter difference:", torch.norm(param_pytorch - param_custom).item())
    print("Exp_avg difference:", torch.norm(exp_avg_pt - exp_avg).item())
    print("Exp_avg_sq difference:", torch.norm(exp_avg_sq_pt - exp_avg_sq).item())
    print("Step difference:", torch.norm(step_t_pt - step_t).item())


if __name__ == '__main__':
    my_tests(globals())
