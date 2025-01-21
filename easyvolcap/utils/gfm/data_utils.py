import torch
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from easyvolcap.utils.console_utils import *


def load_paths(
    data_root,
    camera_names,
    images_dir='images',
    depths_dir='depths',
    ims_pattern='{frame:06d}.jpg',
    frame_sample=[0, None, 1],
    valid_sample=[0, None, 1],
):
    # Get the total number of frames
    n_frames_total = len(os.listdir(join(data_root, images_dir, camera_names[0])))  # NOTE: this is a hacky way to get the number of frames

    # Load image related stuff for reading from disk later
    # If number of images in folder does not match, here we'll get an error
    ims = [[join(data_root, images_dir, cam, ims_pattern.format(frame=i)) for i in range(n_frames_total)] for cam in camera_names]
    if not exists(ims[0][0]):
        ims = [[i.replace('.' + ims_pattern.split('.')[-1], '.JPG') for i in im] for im in ims]
    if not exists(ims[0][0]):
        ims = [[i.replace('.JPG', '.png') for i in im] for im in ims]
    if not exists(ims[0][0]):
        ims = [[i.replace('.png', '.PNG') for i in im] for im in ims]
    if not exists(ims[0][0]):
        ims = [sorted(glob(join(data_root, images_dir, cam, '*')))[:n_frames_total] for cam in camera_names]
    ims = [np.asarray(ims[i])[:min([len(i) for i in ims])] for i in range(len(ims))]  # deal with the fact that some weird dataset has different number of images
    ims = np.asarray(ims)  # V, N
    ims_dir = join(*split(dirname(ims[0, 0]))[:-1])  # logging only

    # TypeError: can't convert np.ndarray of type numpy.str_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
    # MARK: Names stored as np.ndarray
    inds = np.arange(ims.shape[-1])
    if len(valid_sample) != 3: inds = inds[valid_sample]
    else: inds = inds[valid_sample[0]:valid_sample[1]:valid_sample[2]]
    if len(frame_sample) != 3: inds = inds[frame_sample]
    else: inds = inds[frame_sample[0]:frame_sample[1]:frame_sample[2]]
    ims = ims[..., inds]  # these paths are later used for reading images from disk

    # Depth image path preparation
    dps = np.asarray([im.replace(images_dir, depths_dir).replace('.jpg', '.exr').replace('.png', '.exr') for im in ims.ravel()]).reshape(ims.shape)
    if not exists(dps[0, 0]):
        dps = np.asarray([dp.replace('.exr', 'exr') for dp in dps.ravel()]).reshape(dps.shape)
    dps_dir = join(*split(dirname(dps[0, 0]))[:-1])  # logging only

    return ims, ims_dir, dps, dps_dir


def copy_image(
    src_root: str,
    tar_root: str,
    src_images_dir: str = 'image',
    tar_images_dir: str = 'images/00'
):
    # Source and target image directories
    src_img_path = join(src_root, src_images_dir)
    tar_img_path = join(tar_root, tar_images_dir)

    # Create the target directory
    if exists(tar_img_path):
        log(f'{blue(tar_img_path)} already exists, skipping...')
        return
    os.makedirs(dirname(tar_img_path), exist_ok=True)

    # Copy images
    os.system(f'cp -r {src_img_path} {tar_img_path}')
    log(f'copied images from {blue(src_img_path)} to {blue(tar_img_path)}')


def copy_depth(
    src_root: str,
    tar_root: str,
    src_depth_dir: str = 'depth',
    tar_depth_dir: str = 'depths/00'
):
    # Source and target depth directories
    src_dpt_path = join(src_root, src_depth_dir)
    tar_dpt_path = join(tar_root, tar_depth_dir)

    # Create the target directory
    if exists(tar_dpt_path):
        log(f'{blue(tar_dpt_path)} already exists, skipping...')
        return
    os.makedirs(dirname(tar_dpt_path), exist_ok=True)

    # Copy depth maps
    os.system(f'cp -r {src_dpt_path} {tar_dpt_path}')
    log(f'copied depth maps from {blue(src_dpt_path)} to {blue(tar_dpt_path)}')


def link_image(
    src_root: str,
    tar_root: str,
    src_images_dir: str = 'image',
    tar_images_dir: str = 'images/00'
):
    # Source and target image directories
    src_img_path = join(src_root, src_images_dir)
    tar_img_path = join(tar_root, tar_images_dir)

    # Create the target directory
    if exists(tar_img_path):
        log(f'{blue(tar_img_path)} already exists, skipping...')
        return
    os.makedirs(dirname(tar_img_path), exist_ok=True)

    # Link images dir
    os.symlink(relpath(src_img_path, dirname(tar_img_path)), tar_img_path)
    log(f'linked images from {blue(src_img_path)} to {blue(tar_img_path)}')


def link_depth(
    src_root: str,
    tar_root: str,
    src_depth_dir: str = 'depth',
    tar_depth_dir: str = 'depths/00'
):
    # Source and target depth directories
    src_dpt_path = join(src_root, src_depth_dir)
    tar_dpt_path = join(tar_root, tar_depth_dir)

    # Create the target directory
    if exists(tar_dpt_path):
        log(f'{blue(tar_dpt_path)} already exists, skipping...')
        return
    os.makedirs(dirname(tar_dpt_path), exist_ok=True)

    # Link depth maps dir
    os.symlink(relpath(src_dpt_path, dirname(tar_dpt_path)), tar_dpt_path)
    log(f'linked depth maps from {blue(src_dpt_path)} to {blue(tar_dpt_path)}')


def float_to_scientific_no_decimal(num):
    if num == 0:
        return "0e+0"

    # Use scientific notation to format the number
    sci_str = format(num, 'e')
    mantissa, exponent = sci_str.split('e')

    # Remove the trailing zeros
    mantissa = mantissa.rstrip('0').rstrip('.')
    # Remove the decimal point
    mantissa_no_dot = mantissa.replace('.', '')

    # Adjust the exponent
    if '.' in mantissa:
        decimal_places = len(mantissa.split('.')[1])
    else:
        decimal_places = 0
    # Adjust the exponent
    new_exponent = int(exponent) - decimal_places

    return f"{mantissa_no_dot}e{new_exponent:+}"


def save_reproj_error_scatter(
    x: torch.Tensor,
    y: torch.Tensor,
    msk: torch.Tensor,
    filename: str,
    title: str,
    xlabel: str = 'Depth of the first frame',
    ylabel: str = 'Absolute world xyz consistency error',
    s: float = 0.00001,
):
    plt.scatter(
        x[msk].flatten().cpu(),
        y.flatten().cpu(),
        s=s
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.clf()


def save_reproj_error_plot(
    x: List[int],
    y: List[float],
    filename: str,
    title: str = 'World xyz consistency error',
    xlabel: str = 'Frame index',
    ylabel: str = 'Mean world xyz consistency error',
    marker: str = 'o',
):
    plt.scatter(
        x, y, marker=marker
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.clf()
