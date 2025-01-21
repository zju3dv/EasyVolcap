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
        # Check the scene
        scene_name = os.path.basename(scene_root)
        check_root = os.path.join(scene_root, args.check_dir)

        # Choose the game configuration
        if '2077' in scene_root: game = '2077'
        elif 'rdr2' in scene_root: game = 'rdr2'
        elif 'wukong' in scene_root: game = 'wukong'
        else: game = 'default'
        game_cfg = game_cfgs[game]

        # Load the first stage check file
        stage0_result = json.load(
            open(os.path.join(
                check_root, args.check_stage0_dir, args.check_stage0_file
            ))
        )

        # Scene validation flags
        has_colmap = stage0_result['valid_data']
        has_jgcsgt = stage0_result['exist_gt_pose']

        # Skip if the scene is invalid
        if not has_colmap and not has_jgcsgt:
            log(yellow(f'Skipping scene: {blue(scene_root)}'))
            return

        if has_colmap:
            # Convert the COLMAP camera parameters to EasyVolcap
            cmd = [
                'python',
                join(args.script_root, args.script_colmap),
                '--data_root', scene_root,
                '--colmap', stage0_result['sparse_dir'] if \
                    scene_name not in stage0_result['sparse_dir'] else \
                        stage0_result['sparse_dir'][stage0_result['sparse_dir'].index(scene_name) + len(scene_name) + 1:],
                '--output', join(args.check_dir, args.check_evc_dir, 'cameras/colmap/00'),
                '--scale', stage0_result['COLMAP_SCALE']
            ]
            run(cmd)

            # Compute the reprojection error and fuse the depth maps
            # using the COLMAP camera parameters

            # NOTE: Since there will be some COLMAP failure frames, the number of linked raw images will
            # be more than the convert COLMAP cameras. So we need an additional `valid_sample` to filter
            # The `valid_sample` should be a List[int] converted from a List[str]
            valid_sample = [int(i) for i in stage0_result['valid_image_name_list']]

            cmd = [
                'python',
                join(args.script_root, args.script_project),
                '--data_root', scene_root,
                '--easy_root', join(args.check_dir, args.check_evc_dir),
                '--proj_root', join(args.check_dir, args.check_stage1_dir),
                '--output', args.check_stage1_file,
                '--cameras_dir', 'cameras/colmap',
                '--near', stage0_result['near'],
                '--far', stage0_result['far'],
                '--valid_sample'
            ] + valid_sample + [
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

            # Load the reprojection statistics
            colmap_stats = json.load(
                open(
                    join(
                        check_root, args.check_stage1_dir, 'colmap',
                        f'delta{game_cfg.delta}_' + \
                        f'errth{float_to_scientific_no_decimal(args.err_thres)}_' + \
                        f'absth{float_to_scientific_no_decimal(args.abs_thres)}_' + \
                        f'relth{float_to_scientific_no_decimal(args.rel_thres)}',
                        args.check_stage1_file
                    )
                )
            )


        if has_jgcsgt:
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

            # Load the reprojection statistics
            igcsgt_stats = json.load(
                open(
                    join(
                        check_root, args.check_stage1_dir, 'igcsgt',
                        f'delta{game_cfg.delta}_' + \
                        f'errth{float_to_scientific_no_decimal(args.err_thres)}_' + \
                        f'absth{float_to_scientific_no_decimal(args.abs_thres)}_' + \
                        f'relth{float_to_scientific_no_decimal(args.rel_thres)}',
                        args.check_stage1_file
                    )
                )
            )


        # Choose the better result
        if has_colmap and not has_jgcsgt: better_result = colmap_stats
        elif not has_colmap and has_jgcsgt: better_result = igcsgt_stats
        else: better_result = igcsgt_stats if igcsgt_stats['median_abs_err'] < colmap_stats['median_abs_err'] else colmap_stats

        # Save the result of stage 1
        stage1_result = dotdict(
            stage1_result=dotdict(
                valid_data=False,  # default False
            ),
            stage0_result=stage0_result
        )

        # Further checkings
        if better_result['median_abs_err'] < args.max_abs_err and \
          better_result['exclude_ratio'] < args.max_exclude_ratio and \
          len(better_result['valid_image_name_list_now']) > args.min_frames:
            stage1_result['stage1_result'].update(
                dotdict(
                    valid_data=True,
                    valid_image_number_now=len(better_result['valid_image_name_list_now']),
                    cam_path='gt_pose.txt' if better_result['cam_type'] == 'igcsgt' else stage0_result['sparse_dir'],
                    **better_result
                )
            )

        # Save the result of stage 1
        json.dump(
            stage1_result,
            open(join(check_root, args.check_stage1_file), 'w'),
            indent=4
        )

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
    parser.add_argument('--data_root', type=str, default='data/datasets/GameSynthetic/wukong/wukong_xiaoyang_20250109_close_camera_smooth')
    parser.add_argument('--scenes', nargs='+', default=[])
    # Autometic check parameters
    parser.add_argument('--check_dir', type=str, default='check_dir')
    parser.add_argument('--check_evc_dir', type=str, default='easyvolcap')
    parser.add_argument('--check_stage0_dir', type=str, default='')
    parser.add_argument('--check_stage0_file', type=str, default='align_points.json')
    parser.add_argument('--check_stage1_dir', type=str, default='verify')
    parser.add_argument('--check_stage1_file', type=str, default='verify.json')
    # Script paths
    parser.add_argument('--script_root', type=str, default='scripts/gfm/data_verify')
    parser.add_argument('--script_colmap', type=str, default='colmap_to_easyvolcap.py')
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
