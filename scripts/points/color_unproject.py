# this version is too slow, need to improve
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.data_utils import batch_project

import cv2
import open3d as o3d
import numpy as np
import torch

@catch_throw
def main():
    args = dotdict(
        data_dir='./data/selfcap/evc_sm',
        input='pcds',
        output='pcds_color',
        images='images',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    data_dir = args.data_dir
    images_dir = join(data_dir, args.images)
    pcds_dir = join(data_dir, args.input)
    outs_dir = join(data_dir, args.output)
    os.makedirs(outs_dir, exist_ok=True)
    cams = read_camera(data_dir) 

    pcds = sorted(os.listdir(pcds_dir))

    cam_ids = sorted(cams.keys())[2:]
    num_cam = len(cam_ids)
    reserved = ["K", "R", "T", "RT"]
    cams_all = dotdict()
    for cam_id in cam_ids:
        cam = cams[cam_id]
        for key in reserved:
            if key not in cams_all:
                cams_all[key] = []
            cams_all[key].append(torch.from_numpy(cam[key]))
    cams_all = dotdict({key: torch.stack(val) for key, val in cams_all.items()}) # each key is a tensor of shape [num_cam, 3, 3] or [num_cam, 3]

    for i, pcd_name in enumerate((pcds)):
        pcd = join(pcds_dir, pcd_name)
        frame_id = pcd_name.split('.')[0]

        o3d_pcd = o3d.io.read_point_cloud(pcd)
        points = torch.from_numpy(np.array(o3d_pcd.points))

        pixel_coords, depths = batch_project(points.expand(num_cam, -1, -1), cams_all.K, cams_all.RT, return_depth=True)
        points_color = [[] for _ in range(points.shape[0])]

        pbar = tqdm(range(pixel_coords.shape[0]))
        for view_id in pbar:
            xy = pixel_coords[view_id]
            depth = depths[view_id]
            pbar.set_description(f'unproject {view_id+2}_images to {i}_frame pcd')
            image_path = join(images_dir, f'{view_id+2:02d}/{frame_id}.jpg')
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            xy = xy.int()

            # TODO: too slow
            z_buffer = torch.ones(image.shape[:2]) * 20
            pid_buffer = torch.ones(image.shape[:2], dtype=torch.int) * -1
            for point_id, (coord, d) in enumerate(zip(xy, depth)): 
                if 0 <= coord[0] < image.shape[1] and 0 <= coord[1] < image.shape[0]:
                    if d < z_buffer[coord[1], coord[0]]:
                        z_buffer[coord[1], coord[0]] = d
                        pid_buffer[coord[1], coord[0]] = point_id

            # TODO: too slow 
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    point_id = pid_buffer[i, j]
                    if point_id != -1:
                        points_color[point_id].append(image[i, j])
                        
        for id, pt_c in enumerate(points_color):
            if len(pt_c) > 0:
                pt_c_mean = np.mean(np.array(pt_c), axis=0)
                points_color[id] = pt_c_mean / 255.0
            else:
                points_color[id] = [0,0,0]
        o3d_pcd.colors = o3d.utility.Vector3dVector(np.array(points_color))
        out_path = join(outs_dir, f'{pcd_name}')
        o3d.io.write_point_cloud(out_path, o3d_pcd)

if __name__ == '__main__':
    main()
