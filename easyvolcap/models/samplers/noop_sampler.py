import torch
from torch import nn
from easyvolcap.engine import SAMPLERS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import VolumetricVideoModule


@SAMPLERS.register_module()
class NoopSampler(VolumetricVideoModule):
    # Could be uniform in anything, uniform in disparity or weighted uniform
    def __init__(self,
                 network: nn.Module,

                 **kwargs,
                 ):
        super().__init__(network, **kwargs)
        self.forward = self.sample

    def sample(self, ray_o: torch.Tensor, ray_d: torch.Tensor, near: torch.Tensor, far: torch.Tensor, t: torch.Tensor, batch: dotdict):
        return None, None, None, None
