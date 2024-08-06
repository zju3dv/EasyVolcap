"""
Perform camera undistortion on easyvolcap dataset format

colmap image_undistorter --image_path data/selfcap/0330_01/colmap/colmap/images --input_path data/selfcap/0330_01/colmap/colmap/colmap_sparse/0 --output_path data/selfcap/0330_01/colmap/colmap/colmap_dense/0 --blank_pixels 1
"""
from typing import Literal

import cv2
import torch
import kornia
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.easy_utils import read_camera, write_camera
from easyvolcap.utils.data_utils import load_image, save_image, to_tensor
from easyvolcap.utils.undist_utils import colmap_undistort_numpy


def undistort(img: str, out: str, K: np.ndarray, D: np.ndarray, dist_opt_K: bool = True, dist_crop_K: bool = False, backend: Literal['cv2', 'kornia', 'colmap'] = 'cv2'):
    img = load_image(img)

    if backend == 'cv2':
        if dist_opt_K:
            if dist_crop_K:
                new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, img.shape[:2][::-1], 1)
                # dst = cv2.undistort(img, K, D, None, new_K)
                # mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, img.shape[:2][::-1], 5)
                # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            else:
                new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, img.shape[:2][::-1], 0)
            mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_K, img.shape[:2][::-1], 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            # x, y, w, h = roi
            # dst = dst[y:y + h, x:x + w]
        else:
            mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, K, img.shape[:2][::-1], 5)
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            # dst = cv2.undistort(img, K, D)
    elif backend == 'kornia':
        img = to_tensor(img).type(torch.float)
        K = to_tensor(K).type(torch.float)
        D = to_tensor(D).type(torch.float)
        dst = kornia.geometry.calibration.undistort.undistort_image(img.permute(2, 0, 1), K, D[..., 0]).permute(1, 2, 0)
    elif backend == 'colmap':
        dst = colmap_undistort_numpy(img, K, D)[0]
    save_image(out, dst, jpeg_quality=100)


@catch_throw
def main():
    args = dotdict(
        data_root='data/selfcap/0330_01/colmap',
        images_dir='images_dist',
        output_dir='images_undist',
        backend='cv2',  # cv2, colmap, kornia
        dist_opt_K=True,  # remove black edges
        dist_crop_K=False,  # remove black edges
        cameras_dir='',  # root of data root of empty
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    camera_names = sorted(os.listdir(join(args.data_root, args.images_dir)))
    frames = sorted(os.listdir(join(args.data_root, args.images_dir, camera_names[0])))
    cameras = read_camera(join(args.data_root, args.cameras_dir))

    images = [join(args.data_root, args.images_dir, camera, frame) for camera in camera_names for frame in frames]
    outputs = [img.replace(args.images_dir, args.output_dir) for img in images]
    Ks = [cameras[camera].K for camera in camera_names for frame in frames]  # N, 3, 3
    Ds = [cameras[camera].D for camera in camera_names for frame in frames]  # N, 5, 1
    parallel_execution(images, outputs, Ks, Ds, args.dist_opt_K, args.dist_crop_K, args.backend, action=undistort, print_progress=True, sequential=True)


if __name__ == '__main__':
    main()
