import os
import numpy as np
import pandas as pd

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.easy_utils import read_camera, write_camera

from easyvolcap.utils.gfm.game_utils import GameConfig, game_cfgs
from easyvolcap.utils.gfm.camera_utils import load_igcs_pose_v0, load_igcs_pose_v1
from easyvolcap.utils.colmap_utils import write_cameras_binary, write_images_binary, Camera, Image, rotmat2qvec, qvec2rotmat, write_cameras_text, write_images_text


def main():
    args = dotdict(
        data_root='data/datasets/GameSynthetic/wukong/wukong_xietao_250116/250116_012352_375',
        output_dir='check_dir/verify/vggsfm',
        colmap_dir='sparse/0',
        camera_file='gt_pose.txt',
        # https://www.cnblogs.com/xiaohuidi/p/15767477.html
        camera_model=dotdict(choices=['SIMPLE_PINHOLE', 'PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'OPENCV'], default='SIMPLE_PINHOLE'),
        shared_camera=True,

        image_dir='image',
        depth_dir='depth',

        H=1080,
        W=1920,
        fov=None,
        focal=None,
        scale=1.0,
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    # Data roots
    src_root = args.data_root
    tar_root = join(args.data_root, args.output_dir)
    os.makedirs(tar_root, exist_ok=True)

    src_images_dir = join(src_root, args.image_dir)
    tar_images_dir = join(tar_root, f'images/00')

    # Link the images if not exists
    if not os.path.exists(tar_images_dir):
        os.makedirs(dirname(tar_images_dir), exist_ok=True)
        os.symlink(relpath(src_images_dir, dirname(tar_images_dir)), tar_images_dir)

    # Determine the image extension
    image_ext = '.png' if os.path.exists(join(src_images_dir, '000000.png')) else '.jpg'

    src_depths_dir = join(src_root, args.depth_dir)
    tar_depths_dir = join(tar_root, f'depths/00')

    # Link the depths if has depth and not exists
    if os.path.exists(src_depths_dir) and not os.path.exists(tar_depths_dir):
        os.makedirs(dirname(tar_depths_dir), exist_ok=True)
        os.symlink(relpath(src_depths_dir, dirname(tar_depths_dir)), tar_depths_dir)

    # Camera paths
    cam_path = join(src_root, args.camera_file)

    if not os.path.exists(cam_path):
        log(red(f'GT IGCS camera file not found: {blue(cam_path)}'))
        return

    # Choose the game configuration
    game = '2077' if '2077' in src_root else 'rdr2' if 'rdr2' in src_root else 'wukong' if 'wukong' in src_root else 'default'
    game_cfg = game_cfgs[game]
    log(f'using game configuration: {magenta(game_cfg)}')

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

    cameras = {}
    # Prepare the camera parameters
    for i, K in enumerate(Ks):
        # Choose the camera model
        if args.camera_model == 'SIMPLE_PINHOLE': params = [K[0, 0], K[0, 2], K[1, 2]]
        elif args.camera_model == 'PINHOLE': params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        elif args.camera_model == 'SIMPLE_RADIAL': params = [K[0, 0], K[0, 2], K[1, 2], 0.0]
        elif args.camera_model == 'RADIAL': params = [K[0, 0], K[0, 2], K[1, 2], 0.0, 0.0]
        elif args.camera_model == 'OPENCV': params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.0, 0.0, 0.0, 0.0]
        else: raise ValueError(f'Unknown camera model: {args.camera_model}')

        # Create the camera
        camera = Camera(
            id=i,
            model=args.camera_model,
            width=args.W,
            height=args.H,
            params=params
        )
        cameras[i] = camera

        # Exit if shared camera
        if args.shared_camera:
            break

    images = {}
    # Prepare the images
    for i, (R, T) in enumerate(zip(Rs, Ts)):
        qvec = rotmat2qvec(R)
        tvec = T.T
        name = f"{i:06d}{image_ext}"
        camera_id = 0 if args.shared_camera else i

        image = Image(
            id=i,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=[],
            point3D_ids=[]
        )
        images[i] = image

    # Write the cameras
    os.makedirs(join(tar_root, args.colmap_dir), exist_ok=True)
    write_cameras_text(cameras, join(tar_root, args.colmap_dir, 'cameras.txt'))
    write_images_text(images, join(tar_root, args.colmap_dir, 'images.txt'))
    with open(join(tar_root, args.colmap_dir, 'points3D.txt'), 'w') as f:
        f.writelines(['# 3D point list with one line of data per point:\n'])
    # Logging
    log(f'exported camera parameters to {blue(tar_root)}')


if __name__ == "__main__":
    main()
