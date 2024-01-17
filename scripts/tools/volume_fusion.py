"""
Load a easyvolcap model
Perform rendering on all images of a particular frame
Save the rendered rgb and depth value, along with maybe other attributes
Fuse the final rendered depth values as points into one
This function will try to invoke evc programmatically
"""
import torch
import argparse
from os.path import join
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.fcds_utils import voxel_down_sample, remove_outlier
from easyvolcap.utils.data_utils import add_batch, to_cuda, export_pts, export_mesh, export_pcd, to_x
from easyvolcap.utils.math_utils import point_padding, affine_padding, affine_inverse
from easyvolcap.utils.fusion_utils import filter_global_points, depth_geometry_consistency, compute_consistency
from easyvolcap.utils.chunk_utils import multi_gather, multi_scatter
from easyvolcap.utils.cam_utils import compute_camera_similarity

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner


@catch_throw
def main():
    # fmt: off
    import sys
    sys.path.append('.')

    sep_ind = sys.argv.index('--')
    our_args = sys.argv[1:sep_ind]
    evv_args = sys.argv[sep_ind + 1:]
    sys.argv = [sys.argv[0]] + ['-t', 'test'] + evv_args + ['configs=configs/specs/vis.yaml', 'val_dataloader_cfg.dataset_cfg.skip_loading_images=False', 'model_cfg.apply_optcam=True']

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='data/geometry')
    parser.add_argument('--n_srcs', type=int, default=4)
    parser.add_argument('--occ_thresh', type=float, default=0.01)
    parser.add_argument('--geo_abs_thresh', type=float, default=0.5)
    parser.add_argument('--geo_rel_thresh', type=float, default=0.01)
    parser.add_argument('--skip_align', action='store_true')
    parser.add_argument('--skip_density', action='store_true')
    parser.add_argument('--skip_outlier', action='store_true')
    parser.add_argument('--skip_near_far', action='store_true')
    parser.add_argument('--near_far_pad', type=float, default=0.0)
    args = parser.parse_args(our_args)

    # Entry point first, other modules later to avoid strange import errors
    from easyvolcap.scripts.main import test # will do everything a normal user would do
    from easyvolcap.engine import cfg
    from easyvolcap.engine import SAMPLERS
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
    # fmt: on

    runner: VolumetricVideoRunner = test(cfg, dry_run=True)
    runner.load_network()
    runner.model.eval()
    fuse(runner, args)  # pcd is a dotdict with all attributes that we want to retain


def fuse(runner: "VolumetricVideoRunner", args: argparse.Namespace):
    from easyvolcap.dataloaders.datasamplers import get_inds

    dataset = runner.val_dataloader.dataset
    inds = get_inds(dataset)
    nv, nl = inds.shape[:2]
    prefix = 'frame'

    if dataset.closest_using_t:
        nv, nl = nl, nv
        prefix = 'view'
        inds = inds.transpose(0, 1)

    pbar = tqdm(total=nl * nv, desc=f'Fusing rendered RGBD')
    for f in range(nl):
        dpts_cuda = []
        cens_cuda = []
        dirs_cuda = []

        occs_cuda = []
        rgbs_cuda = []

        for v in range(nv):
            # Handle data movement
            batch = dataset[inds[v, f]]  # get the batch data for this view
            batch = add_batch(to_cuda(batch))

            # Running inference
            with torch.inference_mode(), torch.no_grad():
                output = runner.model(batch)  # get everything we need from the model, this performs the actual rendering

            # Get output point clouds
            rgb = batch.rgb[0]
            occ = output.acc_map[0]

            dpt = output.dpt_map[0]
            cen = batch.ray_o[0]
            dir = batch.ray_d[0]

            # Store CUDA depth for later use
            dpts_cuda.append(dpt)  # keep the cuda version for later geometric fusion
            cens_cuda.append(cen)  # keep the cuda version for later geometric fusion
            dirs_cuda.append(dir)  # keep the cuda version for later geometric fusion
            occs_cuda.append(occ)  # keep the cuda version for later geometric fusion
            rgbs_cuda.append(rgb)  # keep the cuda version for later geometric fusion
            pbar.update()

        if dataset.closest_using_t:
            c2ws = dataset.c2ws[f]
            w2cs = dataset.w2cs[f]
            Ks = dataset.Ks[f]
            Hs = dataset.Hs[f]
            Ws = dataset.Ws[f]
        else:
            c2ws = dataset.c2ws[:, f]
            w2cs = dataset.w2cs[:, f]
            Ks = dataset.Ks[: f]
            Hs = dataset.Hs[: f]
            Ws = dataset.Ws[: f]

        _, src_inds = compute_camera_similarity(c2ws, c2ws)  # V, V
        dpts_cuda = torch.stack(dpts_cuda)  # V, HW, 1
        cens_cuda = torch.stack(cens_cuda)  # V, HW, 3
        dirs_cuda = torch.stack(dirs_cuda)  # V, HW, 3

        ptss_cpu = []
        rgbs_cpu = []
        occs_cpu = []

        # Perform depth consistency check and filtering
        for v in range(nv):
            # Prepare source views' information
            src_ind = src_inds[v, 1:1 + args.n_srcs]  # 4,
            dpt_src = dpts_cuda[src_ind]  # 4, HW
            ixt_src = Ks[src_ind]  # 4, 3, 3
            ext_src = affine_padding(w2cs[src_ind])  # 4, 3, 3

            # Prepare reference view's information
            dpt_ref = dpts_cuda[v]  # HW, 1
            ixt_ref = Ks[v]  # 3, 3
            ext_ref = affine_padding(w2cs[v])  # 4, 4

            # Prepare data for computation
            H, W = Hs[v], Ws[v]  # int
            S, HW, C = dpt_src.shape
            dpt_src = dpt_src.view(S, H, W)  # 4, H, W
            dpt_ref = dpt_ref.view(H, W)
            ixt_ref, ext_ref, ixt_src, ext_src = to_cuda([ixt_ref, ext_ref, ixt_src, ext_src])

            depth_est_averaged, photo_mask, geo_mask, final_mask = compute_consistency(
                dpt_ref, ixt_ref, ext_ref, dpt_src, ixt_src, ext_src,
                args.geo_abs_thresh, args.geo_rel_thresh
            )

            # Filter points based on geometry and photometric mask
            ind = final_mask.view(-1).nonzero()  # N, 1
            dpt = multi_gather(depth_est_averaged.view(-1, 1), ind)  # N, 1
            dir = multi_gather(dirs_cuda[v].view(-1, 3), ind)  # N, 3
            cen = multi_gather(cens_cuda[v].view(-1, 3), ind)  # N, 3
            rgb = multi_gather(rgbs_cuda[v].view(-1, 3), ind)  # N, 3
            occ = multi_gather(occs_cuda[v].view(-1, 1), ind)  # N, 1
            pts = cen + dpt * dir  # N, 3

            log(f'View {v}, photo_mask {photo_mask.sum() / photo_mask.numel():.04f}, geometry mask {geo_mask.sum() / geo_mask.numel():.04f}, final mask {final_mask.sum() / final_mask.numel():.04f}, final point count {len(pts)}')

            ptss_cpu.append(pts.detach().cpu())
            rgbs_cpu.append(rgb.detach().cpu())
            occs_cpu.append(occ.detach().cpu())

        # Concatenate per-view depth map and other information
        pts = torch.cat(ptss_cpu, dim=-2).float()  # N, 3
        rgb = torch.cat(rgbs_cpu, dim=-2).float()  # N, 3
        occ = torch.cat(occs_cpu, dim=-2).float()  # N, 1

        # Apply some global filtering
        points = filter_global_points(dotdict(pts=pts, rgb=rgb, occ=occ))
        pts, rgb, occ = points.pts, points.rgb, points.occ
        log(f'Filtered to {len(points.pts)} points globally')

        # Align point cloud with the average camera, which is processed in memory, to make sure the stored files are consistent
        if dataset.use_aligned_cameras and not args.skip_align:  # match the visual hull implementation
            pts = (point_padding(pts) @ affine_padding(dataset.c2w_avg).mT)[..., :3]  # homo

        # Save final fused point cloud back onto the disk
        filename = join(args.result_dir, runner.exp_name, runner.visualizer.save_tag, 'POINT', f'{prefix}{f:04d}.ply')
        export_pts(pts, rgb, filename=filename)
        log(yellow(f'Fused points saved to {blue(filename)}, totally {cyan(pts.numel() // 3)} points'))
    pbar.close()


if __name__ == '__main__':
    main()
