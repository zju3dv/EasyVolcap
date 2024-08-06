import os
import math
import torch
import numpy as np
from enum import Enum, auto

from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.sh_utils import eval_sh, sh_channels, RGB2SH, SH2RGB
from easyvolcap.utils.blend_utils import batch_rodrigues
from easyvolcap.utils.math_utils import torch_inverse_2x2
from easyvolcap.utils.data_utils import to_x, add_batch, load_pts
from easyvolcap.utils.net_utils import make_buffer, make_params, typed
from easyvolcap.utils.gaussian_utils import rgb2sh0, sh02rgb, build_rotation, build_scaling_rotation, strip_symmetric


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=0.01, max_steps=30000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


class STGSRenderMode(Enum):
    stgs = auto()
    diff = auto()
    gsplat = auto()
    taichi = auto()


class STGSModel(nn.Module):
    def __init__(self,
                 xyz,
                 times,
                 colors,
                 init_mode: str = 'naive',
                 init_occ: float = 0.1,
                 init_scale: torch.Tensor = None,
                 init_scale_t: float = 0.1414,
                 sh_deg: int = 3,
                 init_sh_deg: int = 0,
                 bounds: List[List[float]] = None,
                 duration: float = 1.0,
                 xyz_lr_scheduler: dotdict = None,
                 motion_lr_scheduler: dotdict = None,
                 t_lr_scheduler: dotdict = None,
                 opt_cfgs: dotdict = dotdict(noise_clone=False, random_split=True),
                 ):
        super().__init__()

        @torch.jit.script
        def scaling_activation(x):
            return torch.exp(x)
        
        @torch.jit.script
        def scaling_inverse_activation(x):
            return torch.log(torch.clamp(x, 1e-6, 1e6))

        @torch.jit.script
        def opacity_activation(x):
            return torch.sigmoid(x)
        
        @torch.jit.script
        def inverse_opacity_activation(x):
            return torch.logit(torch.clamp(x, 1e-6, 1 - 1e-6))

        self.setup_functions(scaling_activation=scaling_activation, scaling_inverse_activation=scaling_inverse_activation,
                             opacity_activation=opacity_activation, inverse_opacity_activation=inverse_opacity_activation)

        # SH realte configs
        self.active_sh_degree = make_buffer(torch.full((1,), init_sh_deg, dtype=torch.long))  # save them, but need to keep a copy on cpu
        self.cpu_active_sh_degree = self.active_sh_degree.item()
        self.max_sh_degree = sh_deg

        # Set scene bounds and time duration
        self.bounds = make_buffer(torch.as_tensor(bounds, dtype=torch.float))
        self.duration = make_buffer(torch.as_tensor(duration, dtype=torch.float))

        # Initalize trainable parameters
        self.create_from_pcd(xyz, times, colors, init_occ, init_scale, init_scale_t, init_mode)
       
        # Densification related parameters
        self.max_radii2D = make_buffer(torch.zeros(self.get_xyz.shape[0]))
        self.xyz_gradient_accum = make_buffer(torch.zeros((self.get_xyz.shape[0], 1)))
        self.denom = make_buffer(torch.zeros((self.get_xyz.shape[0], 1)))

        # Optimizer related parameters
        self.spatial_scale = torch.norm(self.bounds[1] - self.bounds[0]).item() * 0.5
        self.time_scale = 0.5 * self.duration.item()

        if xyz_lr_scheduler is not None:
            log(yellow_slim('[FDGS] Using xyz learning rate scheduler'))
            xyz_lr_scheduler['lr_init'] *= self.spatial_scale
            xyz_lr_scheduler['lr_final'] *= self.spatial_scale
            self.xyz_scheduler = get_expon_lr_func(**xyz_lr_scheduler)
        else:
            self.xyz_scheduler = None
        if motion_lr_scheduler is not None:
            log(yellow_slim('[FDGS] Using motion learning rate scheduler'))
            motion_lr_scheduler['lr_init'] *= self.spatial_scale
            motion_lr_scheduler['lr_final'] *= self.spatial_scale
            self.motion_scheduler = get_expon_lr_func(**motion_lr_scheduler)
        else:
            self.motion_scheduler = None
        if t_lr_scheduler is not None:
            log(yellow_slim('[FDGS] Using time learning rate scheduler'))
            t_lr_scheduler['lr_init'] *= self.time_scale
            t_lr_scheduler['lr_final'] *= self.time_scale
            self.t_scheduler = get_expon_lr_func(**t_lr_scheduler)
        else:
            self.t_scheduler = None

        self.opt_cfgs = opt_cfgs

        # Perform some model messaging before loading
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        self.post_handle = self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)

    def setup_functions(self,
                        scaling_activation=torch.exp,
                        scaling_inverse_activation=torch.log,
                        opacity_activation=torch.sigmoid,
                        inverse_opacity_activation=torch.logit,
                        rotation_activation=F.normalize,
                        ):
        def build_covariance_from_scaling_rotation(scaling: torch.Tensor, rotation: torch.Tensor):
            L = build_scaling_rotation(scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = getattr(torch, scaling_activation) if isinstance(scaling_activation, str) else scaling_activation
        self.opacity_activation = getattr(torch, opacity_activation) if isinstance(opacity_activation, str) else opacity_activation
        self.rotation_activation = getattr(torch, rotation_activation) if isinstance(rotation_activation, str) else rotation_activation
        self.scaling_inverse_activation = getattr(torch, scaling_inverse_activation) if isinstance(scaling_inverse_activation, str) else scaling_inverse_activation
        self.opacity_inverse_activation = getattr(torch, inverse_opacity_activation) if isinstance(inverse_opacity_activation, str) else inverse_opacity_activation
        self.covariance_activation = build_covariance_from_scaling_rotation

    @property
    def device(self):
        return self.get_xyz.device

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_t(self):
        return self._t
    
    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)
    
    @property
    def get_motion(self):
        return self._motion
    
    def get_marginal_t(self, t: torch.Tensor):
        return torch.exp(-0.5*((t - self.get_t) / self.get_scaling_t)**2)

    def get_covariance(self, scale_mult: float = 1.0):
        return self.covariance_activation(self.get_scaling * scale_mult, self._rotation)

    @property
    def get_max_sh_channels(self):
        return sh_channels[self.max_sh_degree]

    def oneupSHdegree(self):
        changed = False
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.cpu_active_sh_degree = self.active_sh_degree.item()
            changed = True
        
        return changed

    def create_from_pcd(self, xyz: torch.Tensor, times: torch.Tensor, colors, opacities: float = 0.1, scales: torch.Tensor = None, scales_t: float = None, init_mode: str = 'naive'):
        if xyz is None:
            xyz = torch.empty(0, 3, device='cpu')  # by default, init empty gaussian model on CUDA
        # xyz = torch.clamp(xyz, self.bounds[0], self.bounds[1])

        if times is None:
            times = torch.empty(xyz.shape[0], 1, device='cpu')

        features = torch.zeros((xyz.shape[0], 3, self.get_max_sh_channels))
        if colors is not None:
            features[:, :3, 0] = RGB2SH(colors)
        features[:, 3:, 1:] = 0.0

        log(yellow_slim(f'[INIT] NUM POINTS: {xyz.shape[0]}'))

        if scales is None:
            from simple_knn._C import distCUDA2
            dist2 = torch.clamp_min(distCUDA2(xyz.float().cuda()), 0.0000001)
            scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
            # scales = torch.clamp(scales, -10, 1.0)
        else:
            scales = self.scaling_inverse_activation(scales)
        
        # rots = torch.randn((xyz.shape[0], 4))
        rots = torch.zeros((xyz.shape[0], 4))
        rots[:, 0] = 1

        if not isinstance(opacities, torch.Tensor) or len(opacities) != len(xyz):
            opacities = opacities * torch.ones((xyz.shape[0], 1), dtype=torch.float)
        opacities = self.opacity_inverse_activation(opacities)

        motion = torch.zeros((xyz.shape[0], 3), dtype=torch.float, device=xyz.device)

        if scales_t is None:
            scales_t = torch.zeros((xyz.shape[0], 1), dtype=torch.float, device=xyz.device)
        else:
            scales_t = self.scaling_inverse_activation(torch.sqrt(scales_t * torch.ones((xyz.shape[0], 1), dtype=torch.float, device=xyz.device)))

        self._xyz = make_params(xyz)
        self._features_dc = make_params(features[..., :1].mT)
        self._features_rest = make_params(features[..., 1:].mT)
        self._scaling = make_params(scales)
        self._rotation = make_params(rots)
        self._opacity = make_params(opacities)
        self._motion = make_params(motion)
        self._t = make_params(times)
        self._scaling_t = make_params(scales_t)

    @torch.no_grad()
    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Supports loading points and features with different shapes
        if prefix != '' and not prefix.endswith('.'): prefix = prefix + '.'  # special care for when we're loading the model directly
        for name, params in self.named_parameters():
            if f'{prefix}{name}' in state_dict:
                params.data = params.data.new_empty(state_dict[f'{prefix}{name}'].shape)

    @torch.no_grad()
    def _load_state_dict_post_hook(self, module, incompatible_keys):
        # TODO: make this a property that updates the cpu copy on change
        self.cpu_active_sh_degree = self.active_sh_degree.item()

    # def zero_omega(self, mode: str, min_omega: float = 0.15, min_motion: float = 0.3, min_scaling: float = 0.2, max_scaling: float = 0.6, min_opacity: float = 0.7, optimizer: Optimizer = None):
    #     if mode == 'threshold':
    #         mask = torch.sum(torch.abs(self.get_omega), dim=-1) > min_omega
    #     elif mode == 'motion':
    #         mask = torch.sum(torch.abs(self.get_motion[..., 0:3]), dim=-1) > min_motion
    #     else:
    #         raise NotImplementedError
    #     min_scaling_mask = torch.max(self.get_scaling, dim=-1).values > min_scaling
    #     max_scaling_mask = torch.max(self.get_scaling, dim=-1).values < max_scaling
    #     min_opacity_mask = self.get_opacity > min_opacity
    #     mask = torch.logical_and(mask, min_scaling_mask)
    #     mask = torch.logical_and(mask, max_scaling_mask)
    #     mask = torch.logical_and(mask, min_opacity_mask)
    #     new_omage = mask.float()[..., None] * self._omega
    #     self._omega = self.replace_tensor_to_optimizer(new_omage, '_omega', optimizer)
    #     self.omega_mask.set_(mask)
    #     self.omega_mask.grad = None

    def reset_opacity(self, reset_opacity: float = 0.01, optimizer: Optimizer = None):
        log(yellow_slim(f'[RESET OPACITY] REST OPACITY TO {reset_opacity}'))
        new_opacity = torch.min(self._opacity, self.opacity_inverse_activation(torch.ones_like(self._opacity, ) * reset_opacity))
        new_opacity.grad = self._opacity.grad
        self._opacity = self.replace_tensor_to_optimizer(new_opacity, '_opacity', optimizer)

    def reset_t(self, tmin: float = 0.0, tmax: float = 3600.0, optimizer: Optimizer = None):
        log(yellow_slim(f'[RESET T] REST T TO {tmin} - {tmax}'))
        new_t = torch.clamp(self._t, min=tmin, max=tmax)
        new_t.grad = self._t.grad
        self._t = self.replace_tensor_to_optimizer(new_t, '_t', optimizer)

    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str, optimizer: Optimizer):
        optimizable_tensor = None
        for group in optimizer.param_groups:
            if group["name"] == name:
                stored_state = optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensor = group["params"][0]
        if optimizable_tensor is not None:
            return optimizable_tensor
        else:
            log(yellow_slim(f'{name} not found in optimizer'))
            return tensor

    def _prune_optimizer(self, mask: torch.Tensor, optimizer: Optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            attr = getattr(self, group["name"], None)
            if attr is None: continue
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, optimizer: Optimizer):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, optimizer)
        for name, new_params in optimizable_tensors.items():
            setattr(self, name, new_params)
        
    def cat_tensors_to_optimizer(self, tensors_dict: dotdict, optimizer: Optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict.get(group["name"], None)
            if extension_tensor is None: continue
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_motion, optimizer):
        d = dotdict({
            "_xyz": new_xyz,
            "_features_dc": new_features_dc,
            "_features_rest": new_features_rest,
            "_opacity": new_opacities,
            "_scaling": new_scaling,
            "_rotation": new_rotation,
            "_t": new_t,
            "_scaling_t": new_scaling_t,
            "_motion": new_motion,
        })

        optimizable_tensors = self.cat_tensors_to_optimizer(d, optimizer)
        for name, new_params in optimizable_tensors.items():
            setattr(self, name, new_params)
        
    def _split(self, grads, densify_grad_threshold, densify_size_threshold, split_screen_threshold, optimizer: Optimizer = None, N: int = 2):
        # Extract points that satisfy the gradient condition
        n_init_points = self.get_xyz.shape[0]
        device = self.get_xyz.device
        dtype = self.get_xyz.dtype
        # Pad to competible with clone
        padded_grad = torch.zeros((n_init_points,), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # Extract points that satisfy the gradient condition
        high_grads = padded_grad >= densify_grad_threshold
        # Extract points that satisfy the size conditions
        selected_pts_mask = torch.max(self.get_scaling, dim=1).values > densify_size_threshold * self.spatial_scale
        # TODO: Should we consider the size in time?
        # big_time = torch.max(self.get_scaling_t, dim=1).values > densify_t_threshold * self.time_scale
        # selected_pts_mask = torch.logical_or(selected_pts_mask, big_time)
        if split_screen_threshold is not None:
            selected_pts_mask = torch.logical_or(selected_pts_mask, self.max_radii2D > split_screen_threshold)
        selected_pts_mask = torch.logical_and(selected_pts_mask, high_grads)
        n_split = selected_pts_mask.sum().item()
        log(yellow_slim(f'[SPLIT] num points split: {n_split}, num split: {N}.'))
        
        if self.opt_cfgs.random_split:
            # split xyz
            stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
            samples = stds * torch.randn((stds.size(0), 3), device=device, dtype=dtype)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_scaling = self.scaling_inverse_activation(stds / (0.8 * N))
            # split t
            stds = self.get_scaling_t[selected_pts_mask].repeat(N, 1)
            samples = stds * torch.randn((stds.size(0), 1), device=device, dtype=dtype)
            new_t = samples + self.get_t[selected_pts_mask].repeat(N, 1)
            new_scaling_t = self.scaling_inverse_activation(stds / (0.8 * N))
        else:
            raise NotImplementedError

        # split features
        new_motion = self._motion[selected_pts_mask].repeat(N, 1)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_t, new_scaling_t, new_motion, optimizer)
        new_max_radii2D = torch.zeros((n_split * N,), dtype=self.max_radii2D.dtype, device=self.max_radii2D.device)
        self.max_radii2D.set_(torch.cat([self.max_radii2D, new_max_radii2D], dim=0))
        assert self.get_xyz.shape[0] == self.max_radii2D.shape[0]

        prune_mask = torch.cat((selected_pts_mask, torch.zeros((n_split * N,), device=device, dtype=bool)))
        return prune_mask

    def _clone(self, grads, densify_grad_threshold, densify_size_threshold, optimizer: Optimizer = None):
        # Extract points that satisfy the gradient condition
        high_grads = torch.norm(grads, dim=-1) >= densify_grad_threshold
        # Extract points that satisfy the size conditions
        selected_pts_mask = torch.max(self.get_scaling, dim=1).values <= densify_size_threshold * self.spatial_scale
        # TODO: Should we consider the size in time?
        # small_time = torch.max(self.get_scaling_t, dim=1).values <= densify_t_threshold * self.time_scale
        # selected_pts_mask = torch.logical_or(selected_pts_mask, small_time)
        selected_pts_mask = torch.logical_and(selected_pts_mask, high_grads)
        n_clone = selected_pts_mask.sum().item()
        log(yellow_slim(f'[CLONE] num points clone: {n_clone}.'))

        # Clone xyzt
        # Should we just copy? Or should we add some noise to the new points? # NOTE: add noise here
        new_xyz = self._xyz[selected_pts_mask]
        new_t = self._t[selected_pts_mask] # NOTE: just copy
        new_scaling = self._scaling[selected_pts_mask]
        new_scaling_t = self._scaling_t[selected_pts_mask]
        if self.opt_cfgs.noise_clone:
            new_xyz = new_xyz + 0.01 * new_scaling * (torch.rand_like(new_xyz) - 0.5)
            new_t = new_t + 0.01 * new_scaling_t * (torch.rand_like(new_t) - 0.5)
        
        # Clone features
        new_motion = self._motion[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_t, new_scaling_t, new_motion, optimizer)
        new_max_radii2D = torch.zeros((n_clone,), dtype=self.max_radii2D.dtype, device=self.max_radii2D.device)
        self.max_radii2D.set_(torch.cat([self.max_radii2D, new_max_radii2D], dim=0))
        assert self.get_xyz.shape[0] == self.max_radii2D.shape[0]

    def _prune(self, prune_mask, min_opacity: float = None, max_scene_threshold: float = None, max_screen_thresh: float = None, optimizer: Optimizer = None):
        n_before = self.get_xyz.shape[0]
        if prune_mask is None:
            prune_mask = torch.zeros((n_before,), dtype=torch.bool, device=self.get_xyz.device)
            n_mask = 0
        else:
            n_mask = prune_mask.sum().item()
        if min_opacity is not None:
            minocc = (self.get_opacity < min_opacity).squeeze(-1)
            prune_mask = torch.logical_or(prune_mask, minocc)
            n_min_opacity = minocc.sum().item()
        else:
            n_min_opacity = 0
        if max_screen_thresh is not None:
            maxscreen = self.max_radii2D > max_screen_thresh
            prune_mask = torch.logical_or(prune_mask, maxscreen)
            n_max_screen = maxscreen.sum().item()
        else:
            n_max_screen = 0
        if max_screen_thresh is not None:
            maxscene = torch.max(self.get_scaling, dim=1).values > self.spatial_scale * max_scene_threshold
            prune_mask = torch.logical_or(prune_mask, maxscene)
            n_max_scene = maxscene.sum().item()
        else:
            n_max_scene = 0
        self.prune_points(prune_mask, optimizer)
        torch.cuda.empty_cache()
        n_after = self.get_xyz.shape[0]
        log(yellow_slim(f'[PRUNE] ' + 
                        f'num points pruned: {n_before - n_after} ' + 
                        f'num points mask: {n_mask} ' + 
                        f'num points min opacity: {n_min_opacity} ' + 
                        f'num points max screen: {n_max_screen} ' +
                        f'num points max scene: {n_max_scene}.'))
        
    def reset_stats(self):
        device = self.get_xyz.device
        self.xyz_gradient_accum.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))
        self.xyz_gradient_accum.grad = None
        self.denom.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))
        self.denom.grad = None
        self.max_radii2D.set_(torch.zeros((self.get_xyz.shape[0]), device=device))
        self.max_radii2D.grad = None

    def densify_and_prune(self, densify_grad_threshold, min_opacity, densify_size_threshold, max_scene_threshold=None, max_screen_thresh=None, split_screen_threshold=None, optimizer=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        log(yellow_slim(f'[D&P] min grad: {grads.min().item()}, max grad: {grads.max().item()}.'))
        log(yellow_slim(f'[D&P] num points: {self.get_xyz.shape[0]}.'))
        log(yellow_slim(f'[D&P] min radii2D: {self.max_radii2D.min().item()}, max radii2D: {self.max_radii2D.max().item()}.'))
        log(yellow_slim(f'[D&P] min occ: {self.get_opacity.min().item()}, max occ: {self.get_opacity.max().item()}. ' + 
                        f'min scaling: {self.get_scaling.min().item()}, max scaling: {self.get_scaling.max().item()}. ' +
                        f'min t: {self.get_t.min().item()}, max t: {self.get_t.max().item()}. ' +
                        f'min scaling_t: {self.get_scaling_t.min().item()}, max scaling_t: {self.get_scaling_t.max().item()}.'))

        self._clone(grads, densify_grad_threshold, densify_size_threshold, optimizer)
        prune_mask = self._split(grads, densify_grad_threshold, densify_size_threshold, split_screen_threshold, optimizer)
        self._prune(prune_mask, min_opacity, max_scene_threshold, max_screen_thresh, optimizer)
        self.reset_stats()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        xyz_gradient_norm = torch.norm(viewspace_point_tensor.grad.detach()[update_filter, :2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter] += xyz_gradient_norm
        self.denom[update_filter] += 1

    def add_densification_stats_from_grads(self, viewspace_point_tensor_grad, update_filter):
        xyz_gradient_norm = torch.norm(viewspace_point_tensor_grad[..., :2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter] += xyz_gradient_norm
        self.denom[update_filter] += 1

    # def construct_list_of_attributes(self):
    #     l = ['x', 'y', 'z', 'trbf_center', 'trbf_scale', 'nx', 'ny', 'nz'] # 'trbf_center', 'trbf_scale' 
    #     # All channels except the 3 DC
    #     # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
    #     #     l.append('f_dc_{}'.format(i))
    #     # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
    #     #     l.append('f_rest_{}'.format(i))
    #     for i in range(self._motion.shape[1]):
    #         l.append('motion_{}'.format(i))

    #     for i in range(self._features_dc.shape[1]):
    #         l.append('f_dc_{}'.format(i))
    #     # for i in range(self._features_rest.shape[1]):
    #     #     l.append('f_rest_{}'.format(i))
    #     l.append('opacity')
    #     for i in range(self._scaling.shape[1]):
    #         l.append('scale_{}'.format(i))
    #     for i in range(self._rotation.shape[1]):
    #         l.append('rot_{}'.format(i))
    #     for i in range(self._omega.shape[1]):
    #         l.append('omega_{}'.format(i))

    #     return l

    # def save_ply(self, path, gui='STGS'):
    #     # save trained STGS ply
    #     from plyfile import PlyData, PlyElement
    #     os.makedirs(dirname(path), exist_ok=True)

    #     xyz = self._xyz.detach().cpu().numpy()
    #     normals = np.zeros_like(xyz)
    #     f_dc = self._features_dc.detach().cpu().numpy()
    #     opacities = self._opacity.detach().cpu().numpy()
    #     if gui == 'STGS' or gui == 'evc':
    #         scale = self._scaling.detach().cpu().numpy()
    #         t = self._t.detach().cpu().numpy()
    #         scale_t = self._scaling_t.detach().cpu().numpy()
    #     else:
    #         raise NotImplementedError
    #     rotation = self._rotation.detach().cpu().numpy()
    #     motion = self._motion.detach().cpu().numpy()

    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
    #     attributes = np.concatenate((xyz, t, scale_t, normals, motion, f_dc, opacities, scale, rotation, omega), axis=1)
    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(path)

    # def load_ply(self, path: str):
    #     xyz, _, _, scalars = load_pts(path)

    #     # The original gaussian model uses a different activation
    #     xyz = torch.from_numpy(xyz)
    #     rotation = torch.from_numpy(np.concatenate([scalars['rot_{}'.format(i)] for i in range(4)], axis=-1))
    #     scaling = torch.from_numpy(np.concatenate([scalars['scale_{}'.format(i)] for i in range(3)], axis=-1))
    #     scaling = torch.exp(scaling)
    #     scaling = self.scaling_inverse_activation(scaling)
    #     opacity = torch.from_numpy(scalars['opacity'])

    #     # Doing torch.logit here will lead to NaNs
    #     if self.opacity_activation != F.sigmoid and \
    #             self.opacity_activation != torch.sigmoid and \
    #             not isinstance(self.opacity_activation, nn.Sigmoid):
    #         opacity = self.opacity_inverse_activation(opacity)
    #     else:
    #         opacity = torch.logit(opacity)

    #     # Load the SH colors
    #     features_dc = torch.empty((xyz.shape[0], 3, 1))
    #     features_dc[:, 0] = torch.from_numpy(np.asarray(scalars["f_dc_0"]))
    #     features_dc[:, 1] = torch.from_numpy(np.asarray(scalars["f_dc_1"]))
    #     features_dc[:, 2] = torch.from_numpy(np.asarray(scalars["f_dc_2"]))

    #     extra_f_names = [k for k in scalars.keys() if k.startswith("f_rest_")]
    #     extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    #     assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
    #     features_rest = torch.zeros((xyz.shape[0], len(extra_f_names), 1))
    #     for idx, attr_name in enumerate(extra_f_names):
    #         features_rest[:, idx] = torch.from_numpy(np.asarray(scalars[attr_name]))
    #     # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    #     features_rest = features_rest.view(features_rest.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)

    #     state_dict = dotdict()
    #     state_dict._xyz = xyz
    #     state_dict._features_dc = features_dc.mT
    #     state_dict._features_rest = features_rest.mT
    #     state_dict._opacity = opacity
    #     state_dict._scaling = scaling
    #     state_dict._rotation = rotation

    #     self.load_state_dict(state_dict, strict=False)
    #     self.active_sh_degree.data.fill_(self.max_sh_degree)

    def update_learning_rate(self, iter: float, optimizer: Optimizer):
        for param_group in optimizer.param_groups:
            if self.xyz_scheduler is not None and param_group["name"] == "_xyz":
                param_group['lr'] = self.xyz_scheduler(iter)
            if self.motion_scheduler is not None and param_group["name"] == "_motion":
                param_group['lr'] = self.motion_scheduler(iter)
            if self.t_scheduler is not None and param_group["name"] == '_t':
                param_group['lr'] = self.t_scheduler(iter)
