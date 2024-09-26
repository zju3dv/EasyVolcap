"""
Convert camera parameters format from EasyVolcap to nerfstudio
"""

import os
import cv2
import copy
import shutil
import numpy as np
from PIL import Image
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.camera_utils import read_camera


@catch_throw
def main(args):
    args.output_dir = join(args.data_root, args.output_dir)
    args.intri = join(args.data_root, args.intri)
    args.extri = join(args.data_root, args.extri)

    cams = read_camera(args.intri, args.extri)
    assert "basenames" in cams and len(cams["basenames"])
    basenames = sorted(cams["basenames"])
    os.makedirs(args.output_dir, exist_ok=True)

    cameras = {}
    images = {}
    sizes = {}
    frames = os.listdir(join(args.data_root, args.image_dir, basenames[0]))
    frames = sorted([x.split(".")[0] for x in frames])

    b, e, s = args.frame_range
    frames = frames[b:e:s]
    pbar = tqdm(total=len(basenames))

    # Processing cameras
    transform_dict = {"frames": []}
    for cam_id, cam_name in enumerate(basenames):
        # read camera
        cam_dict = cams[cam_name]
        K = cam_dict["K"]
        R = cam_dict["R"]
        T = cam_dict["T"]
        img = cv2.imread(
            join(
                args.data_root, args.image_dir, cam_name, f"{frames[0]}{args.image_ext}"
            )
        )
        h, w = img.shape[:2]

        extrinsic = np.identity(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T[:, 0]
        pose = np.linalg.inv(extrinsic)
        pose[:, 1:3] *= -1  # opencv to opengl

        transform_dict["frames"].append(
            {
                "transform_matrix": pose.tolist(),
                "w": w,
                "h": h,
                "fl_x": K[0, 0],
                "fl_y": K[1, 1],
                "cx": K[0, 2],
                "cy": K[1, 2],
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "camera_model": "OPENCV",
                "camera_label": cam_name,
            }
        )

        pbar.update()

    # Processing images and masks
    pbar = tqdm(total=len(frames))
    for tem_label in frames:
        tranform_frame_dict = copy.deepcopy(transform_dict)
        for frame in tranform_frame_dict["frames"]:
            image_path = join(
                args.image_dir, frame["camera_label"], f"{tem_label}{args.image_ext}"
            )
            mask_path = join(
                args.mask_dir, frame["camera_label"], f"{tem_label}{args.mask_ext}"
            )
            src_image_path = join(args.data_root, image_path)
            src_mask_path = join(args.data_root, mask_path)
            tar_image_path = join(args.output_dir, image_path)
            tar_mask_path = join(args.output_dir, mask_path)

            if args.transparent_bg:
                # save images with alpha channel
                mask = Image.open(src_mask_path)
                mask = mask.convert("L")
                image = Image.open(src_image_path)
                image.putalpha(mask)

                tar_image_path = tar_image_path.replace(args.image_ext, ".png")
                os.makedirs(dirname(tar_image_path), exist_ok=True)
                image.save(tar_image_path)
            else:
                # save images and masks separately
                os.makedirs(dirname(tar_image_path), exist_ok=True)
                shutil.copy(src_image_path, tar_image_path)
                os.makedirs(dirname(tar_mask_path), exist_ok=True)
                shutil.copy(src_mask_path, tar_mask_path)

            frame["file_path"] = "../" + relpath(tar_image_path, args.output_dir)

        os.makedirs(join(args.output_dir, "nerfstudio"), exist_ok=True)
        with open(join(args.output_dir, "nerfstudio", f"{tem_label}.json"), "w") as f:
            json.dump(tranform_frame_dict, f, indent=4)

        pbar.update()


if __name__ == "__main__":
    args = dotdict(
        data_root=f"/mnt/data/home/xuzhen/datasets/renbody/{scene_label}",
        output_dir=f"/mnt/data/home/jinyudong/data/renbody/{scene_label}",
        intri="optimized/intri.yml",
        extri="optimized/extri.yml",
        image_dir="images_calib",
        image_ext=".jpg",
        mask_dir="masks",
        mask_ext=".jpg",
        frame_range=[0, None, 1],
        transparent_bg=False,
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    main(args)
