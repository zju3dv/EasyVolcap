"""
Resample point clouds from one folder to another
"""
import torch
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_pts, export_pts, to_tensor, to_cuda
from easyvolcap.utils.fcds_utils import random_with_features, farthest_with_features, surface_points_features, voxel_surface_down_sample_with_features, voxel_down_sample_and_trace, remove_outlier_with_features, SamplingType

import ast


def parse_multilevel_array(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid multi-level array input: {string}")


@catch_throw
def main():
    args = dotdict(
        # input='data/selfcap/0512_bike/dense_pcds_roma_coarse',
        # output='data/selfcap/0512_bike/coarse_pcds',
        input='/mnt/selfcap/selfcap/0512_bike/dense_pcds_rc',
        output='/mnt/selfcap/selfcap/0512_bike/dense_pcds',
        n_points=120000,  # downsample or upsample to this number of points,
        voxel_size=0.005,
        type=SamplingType.RANDOM_DOWN_SAMPLE.name,
        device='cuda',
        apply_bounds=False,
        bounds=dotdict(default=[[-1.9, -1.0, -2.0], [3.5, 2.8, 6.3]], type=parse_multilevel_array),
        # parser.add_argument('--bounds', default=[[-1.9, -1.0, -2.0], [3.5, 2.8, 6.3]], type=parse_multilevel_array)
        # bounds=[[-3.4786,-0.4362,-1.4715],[2.0149,2.3298,5.6397]],
        std_ratio=0.5,
        frame_sample=[0, None, 1],
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    args.type = SamplingType[args.type]
    args.bounds = torch.as_tensor(args.bounds).to(args.device, non_blocking=True)

    b, e, s = args.frame_sample
    pcds = sorted([f for f in os.listdir(args.input) if f.endswith('.ply') or f.endswith('.obj') or f.endswith('.xyz')])[b:e:s]
    outs = [f.replace(args.input, args.output) for f in pcds]
    outs = [splitext(f)[0] + '.ply' for f in outs]

    for i, (pcd, out) in enumerate(zip(tqdm(pcds), outs)):
        pcd = join(args.input, pcd)
        out = join(args.output, out)
        xyz, rgb = to_cuda(load_pts(pcd)[:2], device=args.device)
        xyz, rgb = xyz[None], rgb[None]
        if args.apply_bounds:
            xyz = xyz[0]
            rgb = rgb[0]
            mask = (xyz >= args.bounds[0]) & (xyz <= args.bounds[1])
            mask = mask.all(dim=-1)  # MARK: SYNC
            xyz = xyz[mask][None]
            rgb = rgb[mask][None]

        if args.type == SamplingType.RANDOM_DOWN_SAMPLE:
            xyz, rgb = random_with_features(xyz, rgb, n_points=args.n_points, std=0.0)  # no extra perturbation
        elif args.type == SamplingType.REMOVE_OUTLIER:
            xyz, rgb = remove_outlier_with_features(xyz, rgb, std_ratio=args.std_ratio)  # no extra perturbation
        elif args.type == SamplingType.FARTHEST_DOWN_SAMPLE:
            xyz, rgb = farthest_with_features(xyz, rgb, n_points=args.n_points)
        elif args.type == SamplingType.VOXEL_DOWN_SAMPLE:
            xyz, rgb = voxel_down_sample_and_trace(xyz, rgb, voxel_size=args.voxel_size)
        elif args.type == SamplingType.SURFACE_DISTRIBUTION:
            xyz, rgb = surface_points_features(xyz, rgb, n_points=args.n_points)
        elif args.type == SamplingType.MARCHING_CUBES_RECONSTRUCTION:
            xyz, rgb = voxel_surface_down_sample_with_features(xyz, rgb, n_points=args.n_points, voxel_size=args.voxel_size)
        elif args.type == SamplingType.NONE:
            xyz, rgb = xyz, rgb
        else:
            raise NotImplementedError

        export_pts(xyz, rgb, filename=out)


if __name__ == '__main__':
    main()
