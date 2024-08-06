import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from os.path import join

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.console_utils import log, run
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.data_utils import export_dotdict, load_image, load_mask, save_image, save_mask
from easyvolcap.utils.parallel_utils import parallel_execution
# fmt: on


def gather_shape(img_path: str):
    img = Image.open(img_path)
    width, height = img.size
    return width, height  # 0 width


def reshape_image(img_path: str, width: int, height: int):
    img = load_image(img_path)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    save_image(img_path, img)


def reshape_mask(msk_path: str, width: int, height: int):
    msk = load_mask(msk_path)
    msk = cv2.resize(msk.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    save_mask(msk_path, msk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/F1_01_000000')
    parser.add_argument('--image_dir', default='images')
    parser.add_argument('--mask_dir', default='bgmt')
    parser.add_argument('--neus_dir', default='neus')
    parser.add_argument('--intri', default='intri.yml')
    parser.add_argument('--extri', default='extri.yml')
    parser.add_argument('--img_name', default='000000.jpg')
    parser.add_argument('--use_min_shape', action='store_true')
    args = parser.parse_args()

    args.image_dir = join(args.data_root, args.image_dir)
    args.mask_dir = join(args.data_root, args.mask_dir)
    args.neus_dir = join(args.data_root, args.neus_dir)
    args.intri = join(args.data_root, args.intri)
    args.extri = join(args.data_root, args.extri)

    # copying images and masks to neus directory
    cam_names = []
    img_paths = []
    msk_paths = []
    neus_img_dir = join(args.neus_dir, 'image')
    neus_msk_dir = join(args.neus_dir, 'mask')
    os.makedirs(neus_img_dir, exist_ok=True)
    os.makedirs(neus_msk_dir, exist_ok=True)
    for i, cam in enumerate(tqdm(sorted(os.listdir(args.image_dir)))):
        ori_img_path = f'{args.image_dir}/{cam}/{args.img_name}'
        ori_msk_path = f'{args.mask_dir}/{cam}/{args.img_name.replace(".jpg", ".png")}'
        # w, h = gather_shape(ori_img_path)
        # if w != 2592 and h != 2048:
        #     continue
        img_path = f'{neus_img_dir}/{i:03d}.jpg'
        msk_path = f'{neus_msk_dir}/{i:03d}.png'
        cam_names.append(cam)  # they just use simple indices for images and masks
        img_paths.append(img_path)
        msk_paths.append(msk_path)
        run(f'cp {ori_img_path} {img_path}', quite=True)  # ? different ext
        run(f'cp {ori_msk_path} {msk_path}', quite=True)

    # reshape images and intrinsics to make them match, we don't want to modify neus' code
    shapes = parallel_execution(img_paths, action=gather_shape)
    shapes = np.array(shapes)
    if args.use_min_shape:
        width = shapes[:, 0].min()  # 0 width
        height = shapes[:, 1].min()
    else:
        width = shapes[:, 0].max()  # 0 width
        height = shapes[:, 1].max()
    ratio = np.zeros_like(shapes, dtype=np.float32)  # evil implicit type conversion...
    ratio[:, 0] = width / shapes[:, 0]  # 0 width
    ratio[:, 1] = height / shapes[:, 1]
    parallel_execution(img_paths, width=width, height=height, action=reshape_image, print_progress=True)  # 0 width
    parallel_execution(msk_paths, width=width, height=height, action=reshape_mask, print_progress=True)  # 0 width

    # converting easymocap camera parameters to neus(idr) format
    # !: removing distortion parameters
    log(f'note that converting to NEUS format will discard distortion information.', 'red')
    camera = read_camera(args.intri, args.extri)
    if 'basenames' in camera:
        del camera['basenames']
    prefix_mat = 'world_mat_{}'
    prefix_scl = 'scale_mat_{}'
    idr_camera = {}
    for i in tqdm(range(len(cam_names))):
        name = prefix_mat.format(i)
        data = camera[cam_names[i]]
        K = data['K']  # 0, width
        K[:2] *= ratio[i, ..., None]  # reshape images and intrinsics, we don't want to modify neus' code
        RT = np.concatenate([data['R'], data['T']], axis=-1)
        idr_camera[name] = K @ RT
        name = prefix_scl.format(i)
        S = np.eye(4)  # 4 by 4 scale correction
        idr_camera[name] = S

    # saving results to disk
    idr_camera_path = join(args.data_root, args.neus_dir, 'cameras.npz')
    log(f'saving camera parameters to {idr_camera_path}')
    export_dotdict(idr_camera, idr_camera_path)

    # preprocess to get normalization matrix
    run(f'python scripts/neus/idr_prepare_normalization.py --source_dir {args.neus_dir}')


if __name__ == '__main__':
    main()
