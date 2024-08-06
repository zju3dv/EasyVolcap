import math
import torch
from typing import List, Union, Optional
from torch.utils.cpp_extension import load
from easyvolcap.utils.console_utils import *


@torch.jit.script
def adam(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_t: torch.Tensor,
    beta1: float,
    beta2: float,
    lr: float,
    eps: float,
):
    # update step
    step_t += 1

    # Decay the first and second moment running average coefficient
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

    step = step_t.item()

    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    step_size = lr / bias_correction1

    bias_correction2_sqrt = math.sqrt(bias_correction2)

    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

    param.addcdiv_(exp_avg, denom, value=-step_size)


module = load(
    name='fused_adam',
    sources=[f'{dirname(__file__)}/src/fused_adam.cu'],
    extra_include_paths=[os.environ.get('CUDA_HOME', '/usr/local/cuda') + "/include"],
    extra_cuda_cflags=["--expt-relaxed-constexpr",
                       "-O2"],
    verbose=True
)


def fused_adam(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_t: torch.Tensor,
    beta1: float,
    beta2: float,
    lr: float,
    eps: float,
):
    step_t += 1
    return module.fused_adam(param, grad, exp_avg, exp_avg_sq, step_t.item(), beta1, beta2, lr, eps)


def _single_tensor_adam(params: List[torch.Tensor],
                        grads: List[torch.Tensor],
                        exp_avgs: List[torch.Tensor],
                        exp_avg_sqs: List[torch.Tensor],
                        max_exp_avg_sqs: List[torch.Tensor],
                        state_steps: List[torch.Tensor],
                        grad_scale: Optional[torch.Tensor],
                        found_inf: Optional[torch.Tensor],
                        *,
                        amsgrad: bool,
                        has_complex: bool,
                        beta1: float,
                        beta2: float,
                        lr: Union[float, torch.Tensor],
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool,
                        differentiable: bool
                        ):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        fused_adam(param, grad, exp_avg, exp_avg_sq, step_t, beta1, beta2, lr, eps)
