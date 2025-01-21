import os
import json
import torch
import numpy as np

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution

from easyvolcap.utils.gfm.game_utils import GameConfig, game_cfgs
from easyvolcap.utils.gfm.data_utils import float_to_scientific_no_decimal


@catch_throw
def main(args):
    # Define the per-scene processing function
    def process_scene(scene_root):
        # Convert the IGCS ground truth camera parameters to EasyVolcap
        cmd = [
            'python',
            join(args.script_root, args.script_igcsgt),
            '--data_root', scene_root,
            '--output', join(args.check_dir, args.check_evc_dir),
            '--src_camera_file', 'gt_pose.txt',
            '--tar_camera_path', 'cameras/igcsgt/00',
            '--H', args.H,
            '--W', args.W,
        ]
        run(cmd)

        # Compute the reprojection error and fuse the depth maps
        # using the IGCS ground truth camera parameters
        cmd = [
            'python',
            join(args.script_root, args.script_project),
            '--data_root', scene_root,
            '--easy_root', join(args.check_dir, args.check_evc_dir),
            '--proj_root', join(args.check_dir, args.check_stage1_dir),
            '--output', args.check_stage1_file,
            '--cameras_dir', 'cameras/igcsgt',
            '--dpt_scale', args.dpt_scale,
            '--dpt_vmaxt', args.dpt_vmaxt,
            '--err_thres', args.err_thres,
            '--abs_thres', args.abs_thres,
            '--rel_thres', args.rel_thres,
        ] + ([
            '--save_figs'
        ] if args.save_figs else []) + ([
            '--save_pcds'
        ] if args.save_pcds else []) + [
            '--vis_views', args.vis_views,
        ] + ([
            '--skip_exclude'
        ] if args.skip_exclude else [])
        run(cmd)

        return


    # Find all scenes
    if len(args.scenes):
        scenes = [os.path.join(args.data_root, f) for f in sorted(os.listdir(args.data_root)) if os.path.isdir(os.path.join(args.data_root, f)) and f in args.scenes]
    else:
        scenes = [os.path.join(args.data_root, f) for f in sorted(os.listdir(args.data_root)) if os.path.isdir(os.path.join(args.data_root, f))]

    # Process each scene
    parallel_execution(scenes, action=process_scene, sequential=True)


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='data/datasets/GameSynthetic/wukong/wukong_xietao_250116')
    parser.add_argument('--scenes', nargs='+', default=[])
    # Autometic check parameters
    parser.add_argument('--check_dir', type=str, default='check_dir')
    parser.add_argument('--check_evc_dir', type=str, default='easyvolcap')
    parser.add_argument('--check_stage1_dir', type=str, default='verify')
    parser.add_argument('--check_stage1_file', type=str, default='verify.json')
    # Script paths
    parser.add_argument('--script_root', type=str, default='scripts/gfm/data_verify')
    parser.add_argument('--script_igcsgt', type=str, default='igcsgt_to_easyvolcap.py')
    parser.add_argument('--script_project', type=str, default='project_points.py')
    parser.add_argument('--H', type=int, default=1080)
    parser.add_argument('--W', type=int, default=1920)
    # Projection parameters
    parser.add_argument('--dpt_scale', type=float, default=1.0)
    parser.add_argument('--dpt_vmaxt', type=float, default=300.)
    parser.add_argument('--err_thres', type=float, default=0.03)
    parser.add_argument('--abs_thres', type=float, default=0.6)
    parser.add_argument('--rel_thres', type=float, default=0.08)
    parser.add_argument('--save_figs', action='store_true')
    parser.add_argument('--save_pcds', action='store_true')
    # Depth fusion parameters
    parser.add_argument('--vis_views', type=int, default=3)
    parser.add_argument('--skip_exclude', action='store_true')
    # Check parameters
    parser.add_argument('--max_abs_err', type=float, default=0.5)
    parser.add_argument('--max_exclude_ratio', type=float, default=0.7)
    parser.add_argument('--min_frames', type=int, default=20)
    args = parser.parse_args()

    # Run the main function
    main(args)
