# Literally return the input as is
import torch
from torch import nn
from easyvolcap.engine import EMBEDDERS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import NoopModule
from easyvolcap.utils.fcds_utils import get_pytorch3d_camera_params
from easyvolcap.utils.data_utils import export_pts, to_x
from pytorch3d.structures import Pointclouds

from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizer,
)

@EMBEDDERS.register_module()
class DepthEmbedder(NoopModule):
    def __init__(self, out_dim=0, normalize=False,**kwargs):
        super().__init__()
        self.out_dim = 1  # no embedding, no output
        self.rasterizer = PointsRasterizer()
        self.normalize = normalize
        
    # calculate the projection of the 3D points onto the image plane, and return the depth value
    def forward(self, inputs: torch.Tensor, batch: dotdict = None):
        # transform the 3D points to the camera coordinate system
        # inputs: B, N, 3
        
        H, W, K, R, T, C = get_pytorch3d_camera_params(batch)
        K, R, T, C = to_x([K, R, T, C], torch.float)
        ndc_pcd = self.rasterizer.transform(Pointclouds(inputs), cameras=PerspectiveCameras(K=K, R=R, T=T, device=inputs.device)).points_padded()  # B, N, 3
        depth = ndc_pcd[..., 2:3] # B, N, 1
        if self.normalize:
            depth = (depth - depth.mean(-2, keepdim=True)) / depth.std(-2, keepdim=True)
        return depth
