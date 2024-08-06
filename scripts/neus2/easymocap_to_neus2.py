# 将单帧数据转换为Nerf格式
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.camera_utils import read_cameras, Undistort, read_camera
from easyvolcap.utils.file_utils import save_json, read_json
from os.path import join
import numpy as np
import cv2
import os
from tqdm import tqdm
import math
from glob import glob


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def convertRT(RT0):
    RT = np.eye(4)
    RT[:3] = RT0
    c2w = np.linalg.inv(RT)
    # convert_mat = np.zeros([4, 4], dtype=RT0.dtype)
    # convert_mat[0, 1] = 1.0
    # convert_mat[1, 0] = 1.0
    # convert_mat[2, 2] = -1.0
    # convert_mat[3, 3] = 1.0
    # c2w = c2w @ convert_mat
    # c2w[0:3, 1] *= -1
    # c2w[0:3, 2] *= -1  # flip the y and z axis
    # # c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
    # # c2w[2, :] *= -1  # flip whole world upside down
    return c2w


# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db):
    da = da/np.linalg.norm(da)
    db = db/np.linalg.norm(db)
    c = np.cross(da, db)
    denom = (np.linalg.norm(c)**2)
    t = ob-oa
    ta = np.linalg.det([t, db, c])/(denom+1e-10)
    tb = np.linalg.det([t, da, c])/(denom+1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db)*0.5, denom


def normalize_cameras(out):
    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0
    totp = [0, 0, 0]
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3, :]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p*w
                totw += w
    totp /= totw
    print(totp)  # the cameras are looking at totp
    # totp = np.array([-1.0, -0.1, -0.1])
    print(totp)  # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= totp
    avglen = 0.
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= len(out['frames'])
    avglen = 1
    print("avg camera distance from origin ", avglen)
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] *= 1./avglen     # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()


if __name__ == '__main__':
    if False:
        data = '/nas/datasets/multi-neuralbody/female-jump'
        out = 'nerf_synthetic/ballet'
        nf = 0
    elif False:
        data = '/nas/datasets/multi-neuralbody/handstand'
        out = 'nerf_synthetic/handstand'
        nf = 0
    elif False:
        data = '/nas/datasets/EasyMocap/static1p'
        out = '/nas/datasets/EasyMocap/static1p-nerf'
        nf = 0
        mask = 'mask'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='./data/renbody/0021_08')
    parser.add_argument('--out', type=str, default='neus2')
    parser.add_argument('--nframe', type=int, default=1000)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--mask_dir', type=str, default='masks')
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--skip_write_image', action='store_true', default=False)
    args = parser.parse_args()

    data = args.path
    out = join(data, args.out)
    nframe, image_dir, mask_dir = args.nframe, args.image_dir, args.mask_dir

    g_H, g_W = None, None
    g_frames = None

    with_mask = True
    if with_mask:
        out_ext = '.png'
        aabb_scale = 1.0
    else:
        out_ext = '.jpg'
        aabb_scale = 4.0
        out += '-back'
    os.makedirs(out, exist_ok=True)
    cameras = read_camera(join(data, 'intri.yml'), join(data, 'extri.yml'))

    subs = cameras['basenames']
    subs.sort()
    cameras = {key: cameras[key] for key in subs}

    for frame in range(0, nframe, args.interval):
        annots = {
            'scale': 0.5,
            'offset': [0.5, 0.5, 0.5],
            'from_na': True,
            'aabb_scale': aabb_scale}
        if not args.skip_write_image:
            annots['frames'] = []
            for sub in tqdm(subs, desc=str(frame)):
                camera = cameras[sub]
                if g_frames is None:
                    g_frames = len(os.listdir(join(data, image_dir, sub)))
                file_path = '{}/{}/{}{}'.format('images', '{:06d}'.format(frame), sub, out_ext)
                imgname = join(data, image_dir, sub, '{:06d}.jpg'.format(frame))
                assert os.path.exists(imgname), imgname
                K, dist = camera['K'].copy(), camera['dist'].copy()
                img = cv2.imread(imgname)
                # b = sharpness(img)
                if with_mask:
                    msknames = glob(join(data, mask_dir, sub, '{:06d}*'.format(frame)))
                    msks = [cv2.imread(mskname, 0) for mskname in msknames]
                    msk = np.zeros_like(img[:, :, 0])
                    for m in msks:
                        m[m > 0] = 255
                        msk = msk | m
                    img = np.dstack([img, msk[..., None]])
                # img = Undistort.image(img, K, dist)
                ori_H, ori_W, _ = img.shape
                if (g_H is None) or (g_W is None):
                    g_H = ori_H
                    g_W = ori_W
                if (g_H != ori_H) or (g_W != ori_W):
                    # perform center crop
                    mode = 'center'
                    if mode == 'center':
                        new_img = np.zeros((g_H, g_W, img.shape[-1]))
                        start_h = (g_H - ori_H) // 2
                        start_w = (g_W - ori_W) // 2
                        if (start_h <= 0) and (start_w <= 0):
                            start_h = abs(start_h)
                            start_w = abs(start_w)
                            new_img = img[start_h:start_h+g_H, start_w:start_w+g_W]
                            K[0, 2] -= start_w
                            K[1, 2] -= start_h
                        elif (start_h > 0) and (start_w > 0):
                            new_img[start_h:start_h+ori_H, start_w:start_w+ori_W] = img
                            K[0, 2] += start_w
                            K[1, 2] += start_h
                        elif (start_h <= 0) and (start_w > 0):
                            start_h = abs(start_h)
                            new_img[:, start_w:start_w+ori_W] = img[start_h:start_h+g_H, :]
                            K[0, 2] += start_w
                            K[1, 2] -= start_h
                        # else (start_h > 0) and (start_w <= 0):
                        else:
                            start_w = abs(start_w)
                            new_img[start_w:start_w+ori_W, :] = img[:, start_w:start_w+g_W]
                            K[0, 2] -= start_w
                            K[1, 2] += start_h
                        img = new_img
                        ori_H, ori_W, _ = img.shape
                    else:
                        raise NotImplementedError
                assert img.shape[0] == g_H
                assert img.shape[1] == g_W
                H, W = int(ori_H * args.ratio), int(ori_W * args.ratio)
                y_ratio, x_ratio = H / ori_H, W / ori_W
                K[0] *= x_ratio
                K[1] *= y_ratio
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)  # MARK: ratio
                outname = join(out, file_path)
                os.makedirs(os.path.dirname(outname), exist_ok=True)
                cv2.imwrite(outname, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                # convert RT
                transform_matrix = convertRT(camera['RT'])
                intrinsic_matrix = np.eye(4)
                intrinsic_matrix[:3, :3] = K
                info = {
                    'file_path': file_path,
                    'transform_matrix': transform_matrix.tolist(),
                    'intrinsic_matrix': intrinsic_matrix.tolist(),
                }
                annots['frames'].append(info)
                annots['w'] = int(g_W * args.ratio)
                annots['h'] = int(g_H * args.ratio)
            # normalize_cameras(annots)
            save_json(join(out,'transforms_'+'{:06d}'.format(frame)+'.json'), annots)
        else:
            data = read_json(join(out,'transforms_'+'{:06d}'.format(frame)+'.json'))
            data.update(annots)
            save_json(join(out,'transforms_'+'{:06d}'.format(frame)+'.json'), data)
        if g_frames and frame + args.interval >= g_frames:
            break