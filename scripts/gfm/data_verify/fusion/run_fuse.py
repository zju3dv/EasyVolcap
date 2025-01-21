import os
import json
import argparse
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.gfm.tsdf_fusion import tsdf_fusion
from easyvolcap.utils.data_utils import load_image, load_depth
from easyvolcap.utils.gfm.game_utils import GameConfig, game_cfgs
from easyvolcap.utils.gfm.camera_utils import load_select_cameras, load_igcs_pose_v0, load_igcs_pose_v1
from easyvolcap.utils.gfm.data_utils import load_paths, link_image, link_depth, float_to_scientific_no_decimal, save_reproj_error_scatter, save_reproj_error_plot


def main(args):
    # Data roots
    src_root = args.data_root
    tar_root = join(args.data_root, args.output)
    cam_path = join(src_root, args.camera_file)

    if not os.path.exists(cam_path):
        log(red(f'GT IGCS camera file not found: {blue(cam_path)}'))
        return

    # Choose the game configuration
    if '2077' in src_root: game = '2077'
    elif 'rdr2' in src_root: game = 'rdr2'
    elif 'wukong' in src_root: game = 'wukong'
    else: game = 'default'
    game_cfg = game_cfgs[game]
    log(f'using game configuration: {magenta(game_cfg)}')

    # Determine the near and far planes
    if args.near is None or args.far is None: near, far = game_cfg.near, game_cfg.far
    else: near, far = args.near, args.far
    log(f'using near: {cyan(near)}, far: {cyan(far)}')

    # Load the camera poses from the dataframe
    df = pd.read_csv(cam_path)
    df.columns = [x.strip() for x in df.columns]

    # Load the camera poses
    if 'gameFrameIndex' not in df.columns:
        cameras, Ks, Rs, Ts, w2cs, c2ws = load_igcs_pose_v0(
            df,
            H=args.H,
            W=args.W,
            fov=args.fov,
            focal=args.focal,
            fov_type=game_cfg.fov_type,
            type=game_cfg.type,
            scale=game_cfg.scale,
        )
    else:
        cameras, Ks, Rs, Ts, w2cs, c2ws = load_igcs_pose_v1(
            df,
            H=args.H,
            W=args.W,
            fov=args.fov,
            focal=args.focal,
            fov_type=game_cfg.fov_type,
            type=game_cfg.type,
            scale=game_cfg.scale,
            delta=game_cfg.delta,
        )

    # Find image paths
    ims = sorted(glob(join(src_root, args.images_dir, '*.png')))
    dps = sorted(glob(join(src_root, args.depths_dir, '*.exr')))
    mesh_save_path = join(tar_root, f"mesh.ply")

    images = []
    depths = []
    # Load images and depth maps
    for image_path, depth_path in zip(ims, dps):
        img = load_image(image_path)
        img = (img[:, :, :3] * 255).astype(np.uint8)
        dpt = load_depth(depth_path)
        dpt = 1 / (dpt * (1 / near - 1 / far) + 1 / far)
        images.append(img)
        depths.append(dpt)

    # Perform TSDF fusion
    tsdf_fusion(
        images=images[:4],
        depths=depths[:4],
        intrinsics=Ks[:4],
        extrinsics=w2cs[:4],
        resolution=args.res,
        mesh_save_path=mesh_save_path,
        depth_scale=args.depth_scale,
        depth_max=args.depth_max,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/datasets/GameSynthetic/wukong/wukong_xietao_250116/250116_012703_254",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="check_dir/verify",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="image",
    )
    parser.add_argument(
        "--depths_dir",
        type=str,
        default="depth",
    )
    parser.add_argument(
        "--camera_file",
        type=str,
        default="gt_pose.txt",
    )
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--far", type=float, default=None)
    parser.add_argument("--H", type=int, default=1080)
    parser.add_argument("--W", type=int, default=1920)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--focal", type=float, default=None)
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_max", type=float, default=1000.0)
    parser.add_argument("--res", type=int, default=512)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
