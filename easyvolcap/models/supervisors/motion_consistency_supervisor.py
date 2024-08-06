import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.console_utils import dotdict
from easyvolcap.utils.loss_utils import ImgLossType, mse, mIoU_loss
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor


@SUPERVISORS.register_module()
class MotionConsistencySupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 motion_consistency: float = 0.0,
                 K: int = 8,
                 _radius: float = 0.1,
                 _scale_xyz: float = 1.0,
                 _scale_t: float = 1.0,
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        self.motion_consistency = motion_consistency
        self.K = K

        # some complex hyperparameters
        self._radius = _radius
        self._scale_xyz = _scale_xyz
        self._scale_t = _scale_t

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        if 'ms3' in output and 't_mask' in output and \
                self.motion_consistency > 0:
            mask = output.t_mask.squeeze(-1)
            xyz = output.xyz[mask]
            ms3 = output.ms3[mask]
            with torch.no_grad():
                from pytorch3d.ops.ball_query import ball_query
                ret = ball_query(xyz[None], xyz[None], K=self.K + 1, radius=self._radius, return_nn=False)
                indices = ret.idx.squeeze(0)[:, 1:]
                valid = (indices > -1).float()
                loss_mask = (valid.sum(1, keepdim=True) > 0).float()
            nbr_ms3 = (ms3[indices] * valid[:, :, None]).sum(1) / (valid.sum(1, keepdim=True) + 1e-6)
            motion_consistency_loss = ((ms3 - nbr_ms3).abs() * loss_mask).mean()
            scalar_stats.mc_loss = motion_consistency_loss
            loss += self.motion_consistency * motion_consistency_loss

        return loss
