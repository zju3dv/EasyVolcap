import os
import torch
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.ray_utils import get_rays
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.image_utils import resize_image, pad_image
from easyvolcap.utils.easy_utils import read_camera, write_camera
from easyvolcap.utils.colmap_utils import qvec2rotmat, rotmat2qvec
from easyvolcap.utils.math_utils import affine_padding, affine_inverse
from easyvolcap.utils.data_utils import to_cuda, to_tensor, to_x, add_batch, load_image, load_depth, export_pts, read_pfm

from easyvolcap.utils.gfm.game_utils import GameConfig, game_cfgs
from easyvolcap.utils.gfm.depth_utils import compute_reproj_error
from easyvolcap.utils.gfm.camera_utils import load_select_cameras, load_igcs_pose_v0, load_igcs_pose_v1
from easyvolcap.utils.gfm.data_utils import load_paths, link_image, link_depth, float_to_scientific_no_decimal, save_reproj_error_scatter, save_reproj_error_plot


@catch_throw
def main(args):
    # Data roots
    src_root = args.data_root
    evc_root = join(args.data_root, args.easy_root)

    # Choose the game configuration
    if '2077' in src_root: game = '2077'
    elif 'rdr2' in src_root: game = 'rdr2'
    elif 'wukong' in src_root: game = 'wukong'
    else: game = 'default'
    game_cfg = game_cfgs[game]
    log(f'using game configuration: {magenta(game_cfg)}')

    # Link the images and depths
    link_image(src_root, evc_root, args.src_images_dir, args.tar_images_dir)
    link_depth(src_root, evc_root, args.src_depths_dir, args.tar_depths_dir)

    # Choose the camera type
    if not exists(join(evc_root, args.cameras_dir)):
        log(red(f'Camera directory not found: {blue(args.cameras_dir)}'))
        return
    cam_type = 'colmap' if 'colmap' in args.cameras_dir else 'igcsgt'

    # Load the camera poses from the dataframe
    camera_names, cameras, Hs, Ws, Ks, Rs, Ts, Ds, Cs, w2cs, c2ws = load_select_cameras(
        evc_root,
        args.cameras_dir,
        view_sample=args.view_sample,
        frame_sample=args.frame_sample,
    )

    # Load paths
    ims, ims_dir, dps, dps_dir = load_paths(
        evc_root,
        camera_names,
        images_dir=args.tar_images_dir if '/' not in args.tar_images_dir else dirname(args.tar_images_dir),
        depths_dir=args.tar_depths_dir if '/' not in args.tar_depths_dir else dirname(args.tar_depths_dir),
        frame_sample=args.frame_sample,
        valid_sample=args.valid_sample,
    )
    log(f'loaded {cyan(ims.shape[0])} cameras, {cyan(ims.shape[1])} frames')

    # Get the indices of the valid samples
    inds = np.arange(len(os.listdir(join(ims_dir, camera_names[0]))))
    if len(args.valid_sample) != 3: inds = inds[args.valid_sample]
    else: inds = inds[args.valid_sample[0]:args.valid_sample[1]:args.valid_sample[2]]

    # Ravel all the loaded stuff
    Hs, Ws, ims, dps, Ks, Rs, Ts, Ds, Cs, w2cs, c2ws = map(
        lambda x: x.reshape(-1, *x.shape[2:]),
            [Hs, Ws, ims, dps, Ks, Rs, Ts, Ds, Cs, w2cs, c2ws]
    )

    # Determine the near and far planes
    if args.near is None or args.far is None: near, far = game_cfg.near, game_cfg.far
    else: near, far = args.near, args.far
    log(f'using near: {cyan(near)}, far: {cyan(far)}')

    # Load depths, images, and rays
    dpts = []
    cens = []
    dirs = []
    rgbs = []
    ixts = []

    w2cs = to_cuda(to_tensor(w2cs)).float()
    c2ws = to_cuda(to_tensor(c2ws)).float()

    for i in tqdm(range(len(Hs)), desc='Loading depths & images & rays'):
        rgb = to_cuda(to_tensor(load_image(ims[i]))).float()[..., :3]  # H, W, 3
        dpt = to_cuda(to_tensor(load_depth(dps[i]))).float() * args.dpt_scale  # H, W, 1

        H, W, K, R, T = Hs[i], Ws[i], Ks[i], Rs[i], Ts[i]
        K, R, T = to_x(to_cuda([K, R, T]), torch.float)

        K[0:1] *= int(W * args.ratio) / W
        K[1:2] *= int(H * args.ratio) / H
        H, W = int(H * args.ratio), int(W * args.ratio)
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = resize_image(rgb, size=(H, W))
        if dpt.shape[0] != H or dpt.shape[1] != W:
            dpt = resize_image(dpt, size=(H, W))

        # Convert the raw depth to linear depth
        dpt = 1 / (dpt * (1 / near - 1 / far) + 1 / far)

        ray_o, ray_d = get_rays(H, W, K, R, T, z_depth=True, correct_pix=False)
        dpts.append(dpt)
        cens.append(ray_o)
        dirs.append(ray_d)
        rgbs.append(rgb)
        ixts.append(K)

    H, W = max([d.shape[-3] for d in dpts]), max([d.shape[-2] for d in dpts])
    dpts = torch.stack([pad_image(i.permute(2, 0, 1), (H, W)).permute(1, 2, 0) for i in dpts])  # V, H, W, 1
    cens = torch.stack([pad_image(i.permute(2, 0, 1), (H, W)).permute(1, 2, 0) for i in cens])  # V, H, W, 3
    dirs = torch.stack([pad_image(i.permute(2, 0, 1), (H, W)).permute(1, 2, 0) for i in dirs])  # V, H, W, 3
    rgbs = torch.stack([pad_image(i.permute(2, 0, 1), (H, W)).permute(1, 2, 0) for i in rgbs])  # V, H, W, 3
    ixts = torch.stack(ixts)  # V, 3, 3

    # Compute the reprojection error between every two adjacent frames
    # Set and create the output directories
    proj_root = join(
        src_root, args.proj_root, cam_type,
        f'delta{game_cfg.delta}_' + \
        f'errth{float_to_scientific_no_decimal(args.err_thres)}_' + \
        f'absth{float_to_scientific_no_decimal(args.abs_thres)}_' + \
        f'relth{float_to_scientific_no_decimal(args.rel_thres)}'
    )
    proj_fig_dir = join(proj_root, f'status')
    proj_pcd_dir = join(proj_root, f'points')
    if args.save_figs:
        os.makedirs(f'{proj_fig_dir}/abs_err', exist_ok=True)
        os.makedirs(f'{proj_fig_dir}/rel_err', exist_ok=True)
    if args.save_pcds:
        os.makedirs(proj_pcd_dir, exist_ok=True)
    os.makedirs(proj_root, exist_ok=True)

    exclude_list = []
    abs_errs = dotdict()
    rel_errs = dotdict()

    for i in range(len(Hs) - 1):
        # Get the depth and camera information of the two adjacent frames
        dpt1, cen1, dir1, ixt1, c2w1, w2c1 = dpts[i+0], cens[i+0], dirs[i+0], ixts[i+0], c2ws[i+0], w2cs[i+0]  # (H, W, 1), (H, W, 3), (H, W, 3), (3, 3), (3, 4), (3, 4)
        dpt2, cen2, dir2, ixt2, c2w2, w2c2 = dpts[i+1], cens[i+1], dirs[i+1], ixts[i+1], c2ws[i+1], w2cs[i+1]  # (H, W, 1), (H, W, 3), (H, W, 3), (3, 3), (3, 4), (3, 4)

        # Compute the depth consistency error
        abs_error, rel_error, wxyz1, wxyz2, mask = compute_reproj_error(
            dpt1, cen1, dir1,
            dpt2, ixt2, c2w2, w2c2,
            H=Hs[i+1], W=Ws[i+1],
            err_thres=args.err_thres,
            dpt_vmaxt=args.dpt_vmaxt
        )

        if args.save_figs:
            # Plot the error in a scatter plot,
            # the x-axis is the depth of the first frame, the y-axis is the error
            # and save the plot to the output directory
            save_reproj_error_scatter(
                dpt1, abs_error, mask,
                filename=f'{proj_fig_dir}/abs_err/frame_{inds[i+0]:06d}_vs_frame_{inds[i+1]:06d}.png',
                title=f'Frame {inds[i+0]:06d} vs Frame {inds[i+1]:06d}',
                ylabel='Absolute world xyz consistency error',
            )
            save_reproj_error_scatter(
                dpt1, rel_error, mask,
                filename=f'{proj_fig_dir}/rel_err/frame_{inds[i+0]:06d}_vs_frame_{inds[i+1]:06d}.png',
                title=f'Frame {inds[i+0]:06d} vs Frame {inds[i+1]:06d}',
                ylabel='Relative world xyz consistency error',
            )

        if args.save_pcds:
            # Save the `wxyz1` in red and `wxyz2` in green to the same file for visualization
            pts = torch.cat([wxyz1[mask], wxyz2], dim=0)  # (N, 3)
            rgb = torch.tensor([[1, 0, 0]]).repeat(len(wxyz1[mask]), 1)  # (N, 3)
            rgb = torch.cat([rgb, torch.tensor([[0, 1, 0]]).repeat(len(wxyz2), 1)], dim=0)  # (N, 3)
            # Save the frame index to the alpha channel
            occ = torch.cat([torch.ones(len(wxyz1[mask]), 1) * i, torch.ones(len(wxyz2), 1) * (i + 1)], dim=0)  # (N, 1)
            export_pts(
                pts, rgb,
                scalars=dotdict(alpha=occ),
                filename=f'{proj_pcd_dir}/frame_{inds[i+0]:06d}_vs_frame_{inds[i+1]:06d}.ply'
            )

        # Compute the mean error
        abs_err = abs_error.mean()
        rel_err = rel_error.mean()
        abs_errs[f'{inds[i+0]:06d}_{inds[i+1]:06d}'] = abs_err.cpu().item()
        rel_errs[f'{inds[i+0]:06d}_{inds[i+1]:06d}'] = rel_err.cpu().item()
        log(f'ABS Frame {inds[i+0]:06d} vs Frame {inds[i+1]:06d}: {cyan(abs_err)}')
        log(f'REL Frame {inds[i+0]:06d} vs Frame {inds[i+1]:06d}: {cyan(rel_err)}')

        if abs_err > args.abs_thres or rel_err > args.rel_thres:
            exclude_list.append(i)

    if args.save_figs:
        # Plot the mean error of every two adjacent frames with
        # x-axis as the frame index and y-axis as the mean error
        save_reproj_error_plot(
            range(len(abs_errs)),
            [abs_errs[f'{inds[i+0]:06d}_{inds[i+1]:06d}'] for i in range(len(abs_errs))],
            filename=f'{proj_root}/abs_err.png'
        )
        save_reproj_error_plot(
            range(len(rel_errs)),
            [rel_errs[f'{inds[i+0]:06d}_{inds[i+1]:06d}'] for i in range(len(rel_errs))],
            filename=f'{proj_root}/rel_err.png'
        )

    # Compute the average error and save the errors to a json file
    # Absolute error
    abs_err_json = dotdict()
    abs_err_json.average = sum(abs_errs.values()) / len(abs_errs)
    abs_err_json.median = np.median(list(abs_errs.values()))
    abs_err_json.frames = abs_errs
    with open(f'{proj_root}/abs_err.json', 'w') as f:
        json.dump(abs_err_json, f, indent=4)
    # Relative error
    rel_err_json = dotdict()
    rel_err_json.average = sum(rel_errs.values()) / len(rel_errs)
    rel_err_json.median = np.median(list(rel_errs.values()))
    rel_err_json.frames = rel_errs
    with open(f'{proj_root}/rel_err.json', 'w') as f:
        json.dump(rel_err_json, f, indent=4)

    # Record the overall statistics
    stats = dotdict(
        cam_type=cam_type,
        median_abs_err=abs_err_json.median,
        median_rel_err=rel_err_json.median,
        average_abs_err=abs_err_json.average,
        average_rel_err=rel_err_json.average,
        exclude_num=len(exclude_list),
        exclude_ratio=len(exclude_list) / len(Hs),
        exclude_list=[f'{inds[i]:06d}' for i in exclude_list]
    )
    stats.valid_image_name_list_now = [
        f'{i:06d}' for i in inds if f'{i:06d}' not in stats.exclude_list
    ]
    # Save the statistics to a json file
    with open(join(proj_root, args.output), 'w') as f:
        json.dump(stats, f, indent=4)
    # Log the excluded frames
    log(f"excluded {cyan(len(exclude_list))} frames: [{cyan(' '.join(stats.exclude_list))}]")
    if args.skip_exclude: log(f'skipping the excluded frames')

    # Fuse the depth maps
    cnt = 0
    xyzs_out = []
    rgbs_out = []
    occs_out = []  # record the frame index

    for v in range(len(cens)):
        # Skip the excluded views
        if v in exclude_list and not args.skip_exclude:
            continue
        # Back-project the points
        xyz = cens[v] + dpts[v] * dirs[v]
        msk = torch.isfinite(xyz).all(dim=-1)
        msk = msk & torch.logical_and(dpts[v] > 0, dpts[v] < args.dpt_vmaxt)[..., 0]
        xyzs_out.append(xyz[msk])
        rgbs_out.append(rgbs[v][msk])
        occs_out.append(torch.full_like(xyz[msk][..., :1], v))
        cnt += 1

    if cnt == 0:
        log(red(f'No valid frames left for fusion'))
        return

    xyz = torch.cat(xyzs_out, dim=-2)
    rgb = torch.cat(rgbs_out, dim=-2)
    occ = torch.cat(occs_out, dim=-2)
    filename = join(proj_root, f'fused_{cam_type}.ply')

    # Downsample the points and export
    downsample_factor = int(cnt // args.vis_views)
    if downsample_factor == 0: downsample_factor = 1
    idx = torch.randperm(xyz.shape[-2])[:xyz.shape[-2] // downsample_factor]
    export_pts(
        xyz[idx],
        rgb[idx],
        scalars=dotdict(alpha=occ[idx]),
        filename=filename
    )
    log(f"fused {cyan(cnt)} frames' points saved to {blue(filename)}, totally {cyan(xyz.numel() // 3)} points")

    return


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()

    # Scene parameters
    parser.add_argument('--data_root', type=str, default='data/datasets/GameSynthetic/wukong/wukong_xiaoyang_20250109_close_camera_smooth/250109_011406_399')
    parser.add_argument('--easy_root', type=str, default='check_dir/easyvolcap')
    parser.add_argument('--proj_root', type=str, default='check_dir/verify')
    parser.add_argument('--output', type=str, default='verify.json')
    parser.add_argument('--cameras_dir', type=str, default='cameras/colmap')
    parser.add_argument('--src_images_dir', type=str, default='image')
    parser.add_argument('--tar_images_dir', type=str, default='images/00')
    parser.add_argument('--src_depths_dir', type=str, default='depth')
    parser.add_argument('--tar_depths_dir', type=str, default='depths/00')
    # Re-projection error parameters
    parser.add_argument('--near', type=float, default=None)
    parser.add_argument('--far', type=float, default=None)
    parser.add_argument('--view_sample', type=int, nargs='+', default=[0, None, 1])
    parser.add_argument('--frame_sample', type=int, nargs='+', default=[0, None, 1])
    parser.add_argument('--valid_sample', type=int, nargs='+', default=[0, None, 1])
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--dpt_scale', type=float, default=1.0)
    parser.add_argument('--dpt_vmaxt', type=float, default=300.)
    parser.add_argument('--err_thres', type=float, default=0.03)
    parser.add_argument('--abs_thres', type=float, default=0.6)
    parser.add_argument('--rel_thres', type=float, default=0.08)
    parser.add_argument('--save_figs', action='store_true')
    parser.add_argument('--save_pcds', action='store_true')
    # Depth fusion parameters
    parser.add_argument('--vis_views', type=int, default=4)
    parser.add_argument('--skip_exclude', action='store_true')
    args = parser.parse_args()

    # Run the main function
    main(args)
