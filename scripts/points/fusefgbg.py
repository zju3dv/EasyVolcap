"""
Resample point clouds from one folder to another
"""
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_pts, export_pts, to_tensor, to_cuda
from easyvolcap.utils.fcds_utils import random, farthest, surface_points, voxel_surface_down_sample, voxel_down_sample, SamplingType
import open3d as o3d

@catch_throw
def main():
    args = dotdict(
        fg='./data/enerf_outdoor/actor1_4/surfs6k',
        bg='./data/enerf_outdoor/actor1_4/bkgd/boost36k/000000.ply',
        out='./data/enerf_outdoor/actor1_4/fusefgbg',
        frame_sample=[0,None,1],
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    
    os.makedirs(args.out, exist_ok=True)

    xyzb, colorsb, normsb, scalarsb = load_pts(args.bg)  # only need xyz for now
    pcds = sorted(os.listdir(args.fg))
    if args.frame_sample[1] is None:
        args.frame_sample[1] = len(pcds)
    pcds = pcds[args.frame_sample[0]:args.frame_sample[1]:args.frame_sample[2]]
    for i, pcd in enumerate(tqdm(pcds)):
        inp_pcd = join(args.fg, pcd)
        # TODO: Add support for preserving vertex properties
        xyz, colors, norms, scalars = load_pts(inp_pcd)  # only need xyz for now
        
        fused_xyz = [xyzb, xyz]
        fused_color = [colorsb, colors]

        log(yellow_slim(f'Fusing point clouds'))
        fused_xyz = np.concatenate(fused_xyz, axis=0)
        fused_color = np.concatenate(fused_color, axis=0)
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(fused_xyz)
        o3d_pcd.colors = o3d.utility.Vector3dVector(fused_color)
        out_pcd = join(args.out, pcd)
        o3d.io.write_point_cloud(out_pcd, o3d_pcd)

if __name__ == '__main__':
    main()
