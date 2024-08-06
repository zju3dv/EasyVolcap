"""
For now, this utility will try to use pycolmap to perform the iterative undistortion implemented by colmap, by calling the apis in the pycolmap repo directly
"""

import cv2
import torch
import pycolmap
import numpy as np
import torch.nn.functional as F
from functools import lru_cache
from easyvolcap.utils.console_utils import log


def colmap_undistort_numpy(img: np.ndarray, K: np.ndarray, D: np.ndarray, blank_pixels: bool = False, device='cpu'):
    dst, Kt = colmap_undistort(torch.from_numpy(img).to(device, non_blocking=True), torch.from_numpy(K).to(device, non_blocking=True), torch.from_numpy(D).to(device, non_blocking=True))
    return dst.cpu().numpy(), Kt.cpu().numpy()


@lru_cache(maxsize=256)
def compute_mapping(H: int, W: int, fx: float, fy: float, cx: float, cy: float, k1: float, k2: float, p1: float, p2: float, k3: float, blank_pixels: bool, device: str = 'cpu'):
    # log(H, W, fx, fy, cx, cy, k1, k2, p1, p2, k3, blank_pixels, device)
    camera = pycolmap.Camera.create(0, 4, 0, W, H)
    undistorted_camera = pycolmap.Camera.create(0, 1, 0, W, H)
    camera.params = [fx, fy, cx, cy, k1, k2, p1, p2]
    undistorted_camera.params = [fx, fy, cx, cy]
    # left_min_x = 1e9
    # left_max_x = -1e9
    # right_min_x = 1e9
    # right_max_x = -1e9
    # top_min_y = 1e9
    # top_max_y = -1e9
    # bottom_min_y = 1e9
    # bottom_max_y = -1e9

    ys = torch.arange(0, H)
    xs = torch.arange(0, W)

    zeros = torch.zeros_like(ys)
    ys1 = torch.stack([zeros + 0.5, ys + 0.5], dim=-1)
    ys2 = torch.stack([zeros + W - 0.5, ys + 0.5], dim=-1)

    zeros = torch.zeros_like(xs)
    xs1 = torch.stack([xs + 0.5, zeros + 0.5], dim=-1)
    xs2 = torch.stack([xs + 0.5, zeros + H - 0.5], dim=-1)

    points1_in_cam = camera.cam_from_img(ys1)
    undistorted_points1 = undistorted_camera.img_from_cam(points1_in_cam)
    points2_in_cam = camera.cam_from_img(ys2)
    undistorted_points2 = undistorted_camera.img_from_cam(points2_in_cam)
    left_min_x = undistorted_points1[:, 0].min()
    left_max_x = undistorted_points1[:, 0].max()
    right_min_x = undistorted_points2[:, 0].min()
    right_max_x = undistorted_points2[:, 0].max()

    points1_in_cam = camera.cam_from_img(xs1)
    undistorted_points1 = undistorted_camera.img_from_cam(points1_in_cam)
    points2_in_cam = camera.cam_from_img(xs2)
    undistorted_points2 = undistorted_camera.img_from_cam(points2_in_cam)
    top_min_y = undistorted_points1[:, 1].min()
    top_max_y = undistorted_points1[:, 1].max()
    bottom_min_y = undistorted_points2[:, 1].min()
    bottom_max_y = undistorted_points2[:, 1].max()

    # for y in range(H):
    #     point1_in_cam = camera.cam_from_img([0.5, y + 0.5])
    #     undistorted_point1 = undistorted_camera.img_from_cam(point1_in_cam)
    #     point2_in_cam = camera.cam_from_img([W - 0.5, y + 0.5])
    #     undistorted_point2 = undistorted_camera.img_from_cam(point2_in_cam)
    #     left_min_x = min(left_min_x, undistorted_point1[0])
    #     left_max_x = max(left_max_x, undistorted_point1[0])
    #     right_min_x = min(right_min_x, undistorted_point2[0])
    #     right_max_x = max(right_max_x, undistorted_point2[0])

    # for x in range(W):
    #     point1_in_cam = camera.cam_from_img([x + 0.5, 0.5])
    #     undistorted_point1 = undistorted_camera.img_from_cam(point1_in_cam)
    #     point2_in_cam = camera.cam_from_img([x + 0.5, H - 0.5])
    #     undistorted_point2 = undistorted_camera.img_from_cam(point2_in_cam)
    #     top_min_y = min(top_min_y, undistorted_point1[1])
    #     top_max_y = max(top_max_y, undistorted_point1[1])
    #     bottom_min_y = min(bottom_min_y, undistorted_point2[1])
    #     bottom_max_y = max(bottom_max_y, undistorted_point2[1])

    min_scale_x = min(cx / (cx - left_min_x), (W - 0.5 - cx) / (right_max_x - cx))
    min_scale_y = min(cy / (cy - top_min_y), (H - 0.5 - cy) / (bottom_max_y - cy))
    max_scale_x = max(cx / (cx - left_max_x), (W - 0.5 - cx) / (right_min_x - cx))
    max_scale_y = max(cy / (cy - top_max_y), (H - 0.5 - cy) / (bottom_min_y - cy))

    scale_x = 1.0 / (min_scale_x * blank_pixels + max_scale_x * (1 - blank_pixels))
    scale_y = 1.0 / (min_scale_y * blank_pixels + max_scale_y * (1 - blank_pixels))

    Wt = int(max(1.0, scale_x * W))
    Ht = int(max(1.0, scale_y * H))
    cxt = Wt / W * cx
    cyt = Ht / H * cy

    x, y = torch.meshgrid(torch.arange(0, Wt, device=device).float(), torch.arange(0, Ht, device=device).float(), indexing='xy')
    u = (x + 0.5 - cxt) / fx
    v = (y + 0.5 - cyt) / fy
    u2 = u * u
    uv = u * v
    v2 = v * v
    r2 = u2 + v2
    radial = k1 * r2 + k2 * r2 * r2
    du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2)
    dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2)
    u = u + du
    v = v + dv
    x = u * fx + cx
    y = v * fy + cy
    src = torch.stack([x, y], dim=-1)

    Kt = torch.as_tensor([
        [fx, 0, cxt],
        [0, fy, cyt],
        [0, 0, 1],
    ]).float().to(device, non_blocking=True)

    return src, Kt


def colmap_undistort(img: torch.Tensor, K: torch.Tensor, D: torch.Tensor, blank_pixels: bool = False):
    # Prepare the camera object
    # img = img.float()
    # K = K.float()
    # D = D.float()

    H, W = img.shape[-3:-1]
    D = D.ravel()[..., None]
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item(), D[0, 0].item(), D[1, 0].item(), D[2, 0].item(), D[3, 0].item(), D[4, 0].item()
    device = img.device.type

    src, Kt = compute_mapping(H, W, fx, fy, cx, cy, k1, k2, p1, p2, k3, blank_pixels, device=device)

    # img_tensor = torch.from_numpy(img).float()
    if img.device.type == 'cuda':
        src = src / torch.as_tensor([W, H], dtype=torch.float32).to(device, non_blocking=True) * 2 - 1
        dst = F.grid_sample(img.float().permute(2, 0, 1)[None], src[None], mode='bilinear', align_corners=False)[0].permute(1, 2, 0)
    else:
        # src = (src + 1) / 2 * torch.as_tensor([W, H], dtype=torch.float32)

        img = img.numpy()
        src = src.numpy()
        dst = cv2.remap(img, src[..., 0], src[..., 1], cv2.INTER_LINEAR)
        dst = torch.from_numpy(dst)
    return dst, Kt
