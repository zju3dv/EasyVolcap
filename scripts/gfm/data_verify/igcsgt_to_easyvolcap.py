import os
import numpy as np
import pandas as pd

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.easy_utils import read_camera, write_camera

from easyvolcap.utils.gfm.game_utils import GameConfig, game_cfgs
from easyvolcap.utils.gfm.camera_utils import load_igcs_pose_v0, load_igcs_pose_v1


def main():
    args = dotdict(
        data_root='data/datasets/GameSynthetic/wukong/wukong_xietao_250116/250116_012352_375',
        output='check_dir/easyvolcap',
        src_camera_file='gt_pose.txt',
        tar_camera_path='cameras/igcsgt/00',
        H=1080,
        W=1920,
        fov=None,
        focal=None,
        scale=1.0,
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    # Data roots
    src_root = args.data_root
    tar_root = join(args.data_root, args.output)

    # Camera paths
    src_cam_path = join(src_root, args.src_camera_file)
    tar_cam_path = join(tar_root, args.tar_camera_path)

    if not os.path.exists(src_cam_path):
        log(red(f'GT IGCS camera file not found: {blue(src_cam_path)}'))
        return

    # Choose the game configuration
    if '2077' in src_root: game = '2077'
    elif 'rdr2' in src_root: game = 'rdr2'
    elif 'wukong' in src_root: game = 'wukong'
    else: game = 'default'
    game_cfg = game_cfgs[game]
    log(f'using game configuration: {magenta(game_cfg)}')

    # Load the camera poses from the dataframe
    df = pd.read_csv(src_cam_path)
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

    # Export the camera parameters
    write_camera(cameras, tar_cam_path)
    log(f'exported camera parameters to {blue(tar_cam_path)}')


if __name__ == "__main__":
    main()
