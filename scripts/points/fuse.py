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
        input='./data/selfcap/evc_sm/pcds',
        output='./data/selfcap/evc_sm/fused.ply',
        downsample_voxel_size=0.01,
        remove_outlier=False,
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    pcds = sorted(os.listdir(args.input))
    output = args.output
    fused_xyz = []
    fused_color = []
    for i, pcd in enumerate(tqdm(pcds)):
        pcd = join(args.input, pcd)
        # TODO: Add support for preserving vertex properties
        xyz, colors, norms, scalars = load_pts(pcd)  # only need xyz for now
        fused_xyz.append(xyz)
        fused_color.append(colors)
    
    log(yellow_slim(f'Fusing point clouds'))
    fused_xyz = np.concatenate(fused_xyz, axis=0)
    fused_color = np.concatenate(fused_color, axis=0)
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(fused_xyz)
    o3d_pcd.colors = o3d.utility.Vector3dVector(fused_color)
    o3d.io.write_point_cloud(output.replace('fused', 'fused_orig'), o3d_pcd)

    if args.remove_outlier:
        # o3d_pcd, ind = o3d_pcd.remove_radius_outlier(nb_points=50, radius=0.05)
        # o3d.io.write_point_cloud(output.replace('fused', 'fused_filter'), o3d_pcd) 
        cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        o3d_pcd = o3d_pcd.select_by_index(np.array(ind))
        log(yellow_slim(f'filter {fused_xyz.shape[0]} -> {len(ind)}'))

    downsample_voxel_size = args.downsample_voxel_size
    if downsample_voxel_size is not None:
        target_voxel_size = downsample_voxel_size # 1cm一个点，应该够了
        o3d_pcd = o3d_pcd.voxel_down_sample(target_voxel_size)

    log(yellow_slim(f'exporting {len(o3d_pcd.points)} points to {output}'))
    o3d.io.write_point_cloud(output, o3d_pcd)

if __name__ == '__main__':
    main()
