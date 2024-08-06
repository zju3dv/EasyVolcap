import torch
import itertools
from torch import nn
from typing import Iterator, Tuple, Mapping, Dict

from torch.optim import Adam, AdamW, SGD, LBFGS, Optimizer
from easyvolcap.engine import OPTIMIZERS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import *

OPTIMIZERS.register_module()(Adam)
OPTIMIZERS.register_module()(AdamW)
OPTIMIZERS.register_module()(SGD)
OPTIMIZERS.register_module()(LBFGS)


@OPTIMIZERS.register_module()
class MyFusedAdam(Adam):
    def step(self, closure=None):
        """Perform a single optimization step.

        Will disrespect weight decay, but significantly reduce kernel launches

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            from easyvolcap.utils.adam_utils import _single_tensor_adam
            _single_tensor_adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=getattr(self, "amsgrad", False),
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=getattr(self, "maximize", False),
                capturable=getattr(self, "capturable", False),
                differentiable=getattr(self, "differentiable", False),
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps
    ):
        # Older version of PyTorch doesn't have this method
        has_complex = False
        for p in group['params']:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state['step'] = (
                        torch.tensor(0.0, dtype=torch.float32)
                    )
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                state_steps.append(state['step'])
        return has_complex


@OPTIMIZERS.register_module()
def ConfigurableOptimizer(named_params: Iterator[Tuple[str, nn.Parameter]],

                          # Default parameters
                          lr: float = 5e-3,
                          eps: float = 1e-15,
                          weight_decay: float = 0.0,

                          # Special parameters
                          lr_table: dotdict = dotdict(),  # empty special learning rate table
                          eps_table: dotdict = dotdict(),  # empty table
                          weight_decay_table: dotdict = dotdict(),  # empty table

                          fused: bool = None,
                          foreach: bool = None,

                          optimizer_cfg: dotdict = dotdict(type=Adam.__name__),
                          ) -> Optimizer:
    if isinstance(named_params, Iterator):
        first = next(named_params)
        if isinstance(first, Tuple):
            named_params = itertools.chain([first], named_params)
        elif isinstance(first, nn.Parameter):
            log(yellow(f'Passed in a list of parameters, assuming they are named sequentially.'))
            named_params = {str(i): first for i, first in enumerate(named_params)}.items()
        else:
            raise NotImplementedError
    elif isinstance(named_params, Dict):
        named_params = named_params.items()
    else:
        raise NotImplementedError

    lr_line = dotdict()
    lr_line.lr = lr
    lr_line.eps = eps
    lr_line.weight_decay = weight_decay
    if lr_line: log('Starting learning rate config:', line(lr_line))

    lr_line = dotdict()
    if len(lr_table): lr_line.lr = lr_table
    if len(eps_table): lr_line.eps = eps_table
    if len(weight_decay_table): lr_line.weight_decay = weight_decay_table
    if lr_line: log('Special learning rate config:', line(lr_line))

    # This is resulting in a lot of parameter groups, might reach cuda launch queue depth limit for optimization
    # One option is to consider settings the same type of parameters in the same group, but this might not be the intended behavior for the user
    # Another is to only perform optimization step on the parameters that received gradient
    param_groups = []
    for key, value in named_params:
        if not value.requires_grad:
            continue  # skip non-optimizable paramters
        v_lr = lr
        v_eps = eps
        v_weight_decay = weight_decay
        keys = key.split('.')
        for item in keys:
            if item in lr_table:
                v_lr = lr_table[item]
                break
        for item in keys:
            if item in eps_table:
                v_eps = eps_table[item]
                break
        for item in keys:
            if item in weight_decay_table:
                v_weight_decay = weight_decay_table[item]
                break
        param_groups.append(
            dotdict(
                params=[value],
                lr=v_lr,
                eps=v_eps,
                weight_decay=v_weight_decay,
                name=key,
                fused=fused,
                foreach=foreach,
            )
        )

    if not len(param_groups):
        log(red('optimizer got an empty parameter list, assume you\'re testing'))
        return None

    return OPTIMIZERS.build(optimizer_cfg, params=param_groups)
