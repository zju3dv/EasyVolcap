import os
import torch
import torch.nn.functional as F


def compute_reproj_error(
    dpt1: torch.Tensor,  # (H, W, 1)
    cen1: torch.Tensor,  # (H, W, 3)
    dir1: torch.Tensor,  # (H, W, 3)
    dpt2: torch.Tensor,  # (H, W, 1)
    ixt2: torch.Tensor,  # (3, 3)
    c2w2: torch.Tensor,  # (4, 4)
    w2c2: torch.Tensor,  # (4, 4)
    H: int = 1080,
    W: int = 1920,
    err_thres: float = 0.03,
    dpt_vmaxt: float = 300.,
):
    # Lift the depth of the first frame to the world space
    wxyz1 = cen1 + dpt1 * dir1  # (H, W, 3)

    # Project the 3D points to the second frame
    iuvd2 = ixt2[None, None] @ w2c2[None, None] @ torch.cat([wxyz1, torch.ones_like(dpt1)], dim=-1)[..., None]  # (H, W, 3, 1)
    iuvd2 = iuvd2[..., 0]  # (H, W, 3)
    iuv2 = iuvd2[..., :2] / (iuvd2[..., 2:] + 1e-8)  # (H, W, 2)

    # Check if the projected points are within the image boundary
    u2, v2 = iuv2.chunk(2, dim=-1)  # (H, W, 1), (H, W, 1)
    mask = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)  # (H, W, 1)
    mask = mask[..., 0]  # (H, W)
    mask = mask & ((dpt1 > 0) & (dpt2 > 0) & (dpt1 < dpt_vmaxt) & (dpt2 < dpt_vmaxt))[..., 0]  # (H, W)

    # Sample the depth of the second frame
    grid = torch.cat([u2 / W * 2 - 1, v2 / H * 2 - 1], dim=-1)  # (H, W, 2)
    grid = grid[mask][None, None]  # (1, 1, N, 2)
    dpt2_ = F.grid_sample(dpt2.permute(2, 0, 1)[None], grid, align_corners=False)[0].permute(1, 2, 0)[0]  # (N, 1)

    # Lift the depth of the second frame to the world space
    cxyz2 = torch.inverse(ixt2)[None] @ torch.cat([iuv2, torch.ones_like(dpt2)], dim=-1)[mask][..., None]  # (N, 3, 1)
    cxyz2 = cxyz2 * dpt2_[..., None]  # (N, 3, 1)
    wxyz2 = c2w2[None, :3, :3] @ cxyz2 + c2w2[None, :3, 3:]  # (N, 3, 1)
    wxyz2 = wxyz2[..., 0]  # (N, 3)

    # Compute the depth consistency error
    abs_error = (torch.norm(wxyz1[mask] - wxyz2, dim=-1) > err_thres).float()  # (N)
    rel_error = abs_error / dpt1[mask][..., 0]  # (N)
    # error = torch.norm(wxyz1[mask] - wxyz2, dim=-1)  # (N)
    # abs_error = (error > err_thres).float()  # (N)
    # rel_error = error / dpt1[mask][..., 0]  # (N)

    return abs_error, rel_error, wxyz1, wxyz2, mask
