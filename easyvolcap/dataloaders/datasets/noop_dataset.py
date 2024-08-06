import torch
from torch.utils.data import Dataset
from typing import List, Optional

from easyvolcap.engine import DATASETS, cfg, args
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.math_utils import normalize, affine_inverse

from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset
from easyvolcap.models.cameras.optimizable_camera import OptimizableCamera


@DATASETS.register_module()
class NoopDataset(Dataset):
    def __init__(self,
                 frame_sample: List[float],
                 view_sample: List[float],
                 closest_using_t: bool = False,
                 near: float = 0.02,
                 far: float = 100.0,
                 bounds: List[List[float]] = torch.as_tensor([[-5, -5, -5], [5, 5, 5]]),
                 render_ratio: float = 1.0,
                 focal_ratio: float = 1.0,
                 duration: float = 1.0,

                 imbound_crop: bool = True,
                 use_objects_priors: bool = False,
                 **kwargs):
        self.view_sample = view_sample
        self.frame_sample = frame_sample
        self.closest_using_t = closest_using_t
        self.n_views = (view_sample[1] - view_sample[0]) // view_sample[2] if view_sample[1] is not None else 1
        self.n_latents = (frame_sample[1] - frame_sample[0]) // frame_sample[2] if frame_sample[1] is not None else 1
        self.n_frames_total = self.n_latents
        self.n_view_total = self.n_views

        self.near = near
        self.far = far
        self.bounds = torch.as_tensor(bounds, dtype=torch.float)
        self.render_ratio = render_ratio
        self.focal_ratio = focal_ratio

        self.imbound_crop = imbound_crop
        self.use_objects_priors = use_objects_priors

        self.duration = duration
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        return self.n_latents * self.n_views

    @property
    def frame_min(self): return VolumetricVideoDataset.frame_min.fget(self)
    @property
    def frame_max(self): return VolumetricVideoDataset.frame_max.fget(self)
    @property
    def frame_int(self): return VolumetricVideoDataset.frame_int.fget(self)
    @property
    def frame_range(self): return VolumetricVideoDataset.frame_range.fget(self)

    @property
    def view_min(self): return VolumetricVideoDataset.view_min.fget(self)
    @property
    def view_max(self): return VolumetricVideoDataset.view_max.fget(self)
    @property
    def view_int(self): return VolumetricVideoDataset.view_int.fget(self)
    @property
    def view_range(self): return VolumetricVideoDataset.view_range.fget(self)

    def t_to_frame(self, t): return VolumetricVideoDataset.t_to_frame(self, t)

    def frame_to_t(self, frame_index): return VolumetricVideoDataset.frame_to_t(self, frame_index)

    def frame_to_latent(self, frame_index): return VolumetricVideoDataset.frame_to_latent(self, frame_index)

    def camera_to_v(self, camera_index): return VolumetricVideoDataset.camera_to_v(self, camera_index)

    def v_to_camera(self, v): return VolumetricVideoDataset.v_to_camera(self, v)

    def camera_to_view(self, camera_index): return VolumetricVideoDataset.camera_to_view(self, camera_index)

    def get_bounds(self, latent_index): return VolumetricVideoDataset.get_bounds(self, latent_index)

    @staticmethod
    def scale_ixts(output: dotdict, ratio: float): return VolumetricVideoDataset.scale_ixts(output, ratio)

    @staticmethod
    def crop_ixts_bounds(output: dotdict): return VolumetricVideoDataset.crop_ixts_bounds(output)

    def virtual_to_physical(self, latent_index): return latent_index

    def physical_to_virtual(self, latent_index): return latent_index

    def get_viewer_batch(self, *args, **kwargs): return VolumetricVideoDataset.get_viewer_batch(self, *args, **kwargs)
