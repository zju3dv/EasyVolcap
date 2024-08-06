import os
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def get_Ps(cameras, number_of_cameras):
    Ps = []
    for i in range(0, number_of_cameras):
        P = cameras['world_mat_%d' % i][:3, :].astype(np.float64)
        Ps.append(P)
    return np.array(Ps)


# Gets the fundamental matrix that transforms points from the image of camera 2, to a line in the image of
# camera 1
def get_fundamental_matrix(P_1, P_2):
    P_2_center = np.linalg.svd(P_2)[-1][-1, :]
    epipole = P_1 @ P_2_center
    epipole_cross = np.zeros((3, 3))
    epipole_cross[0, 1] = -epipole[2]
    epipole_cross[1, 0] = epipole[2]

    epipole_cross[0, 2] = epipole[1]
    epipole_cross[2, 0] = -epipole[1]

    epipole_cross[1, 2] = -epipole[0]
    epipole_cross[2, 1] = epipole[0]

    F = epipole_cross @ P_1 @ np.linalg.pinv(P_2)
    return F


# Given a point (curx,cury) in image 0, get the  maximum and minimum
# possible depth of the point, considering the second image silhouette (index j)
def get_min_max_d(curx, cury, P_j, silhouette_j, P_0, Fj0, j, num_of_matching_trials=100000):
    # randomly choose some points from the silhouette
    indss = np.random.choice(silhouette_j.shape[1], (min(num_of_matching_trials, silhouette_j.shape[1]),))
    silhouette_j = silhouette_j[:, indss]

    # transfer point to line using the fundamental matrix:
    cur_l_1 = Fj0 @ np.array([curx, cury, 1.0]).astype(np.float32)
    cur_l_1 = cur_l_1 / np.linalg.norm(cur_l_1[:2])

    # Distances of the silhouette points from the epipolar line:
    dists = np.abs(silhouette_j.T @ cur_l_1)
    relevant_matching_points_1 = silhouette_j[:, dists < 0.7]

    if relevant_matching_points_1.shape[1] == 0:
        return (0.0, 0.0)
    X = cv2.triangulatePoints(P_0, P_j, np.tile(np.array([curx, cury]).astype(np.float32),
                                                (relevant_matching_points_1.shape[1], 1)).T,
                              relevant_matching_points_1[:2, :])
    depths = P_0[2] @ (X / X[3])
    reldepth = depths >= 0
    depths = depths[reldepth]
    if depths.shape[0] == 0:
        return (0.0, 0.0)

    min_depth = depths.min()
    max_depth = depths.max()

    return min_depth, max_depth

# get all fundamental matrices that trasform points from camera 0 to lines in Ps


def get_fundamental_matrices(P_0, Ps):
    Fs = []
    for i in range(0, Ps.shape[0]):
        F_i0 = get_fundamental_matrix(Ps[i], P_0)
        Fs.append(F_i0)
    return np.array(Fs)


def get_all_mask_points(masks_dir):
    mask_paths = sorted(glob_imgs(masks_dir))
    mask_points_all = []
    mask_ims = []
    for path in mask_paths:
        img = mpimg.imread(path)
        if img.ndim == 2:
            img = img[..., None]
        cur_mask = img.max(axis=2) > 0.5
        mask_points = np.where(img.max(axis=2) > 0.5)
        xs = mask_points[1]
        ys = mask_points[0]
        mask_points_all.append(np.stack((xs, ys, np.ones_like(xs))).astype(np.float32))
        mask_ims.append(cur_mask)
    return mask_points_all, np.stack(mask_ims)


def refine_visual_hull(masks, Ps, scale, center):
    num_cam = masks.shape[0]
    GRID_SIZE = 100
    MINIMAL_VIEWS = 45  # Fitted for DTU, might need to change for different data.
    im_height = masks.shape[1]
    im_width = masks.shape[2]
    xx, yy, zz = np.meshgrid(np.linspace(-scale, scale, GRID_SIZE), np.linspace(-scale, scale, GRID_SIZE),
                             np.linspace(-scale, scale, GRID_SIZE))
    points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()))
    points = points + center[:, np.newaxis]
    appears = np.zeros((GRID_SIZE * GRID_SIZE * GRID_SIZE, 1))
    for i in range(num_cam):
        proji = Ps[i] @ np.concatenate((points, np.ones((1, GRID_SIZE * GRID_SIZE * GRID_SIZE))), axis=0)
        depths = proji[2]
        proj_pixels = np.round(proji[:2] / depths).astype(np.int32)
        relevant_inds = np.logical_and(proj_pixels[0] >= 0, proj_pixels[1] < im_height)
        relevant_inds = np.logical_and(relevant_inds, proj_pixels[0] < im_width)
        relevant_inds = np.logical_and(relevant_inds, proj_pixels[1] >= 0)
        relevant_inds = np.logical_and(relevant_inds, depths > 0)
        relevant_inds = np.where(relevant_inds)[0]

        cur_mask = masks[i] > 0.5
        relmask = cur_mask[proj_pixels[1, relevant_inds], proj_pixels[0, relevant_inds]]
        relevant_inds = relevant_inds[relmask]
        appears[relevant_inds] = appears[relevant_inds] + 1

    final_points = points[:, (appears >= MINIMAL_VIEWS).flatten()]
    centroid = final_points.mean(axis=1)
    normalize = final_points - centroid[:, np.newaxis]

    return centroid, np.sqrt((normalize ** 2).sum(axis=0)).mean() * 3, final_points.T

# the normaliztion script needs a set of 2D object masks and camera projection matrices (P_i=K_i[R_i |t_i] where [R_i |t_i] is world to camera transformation)


def get_normalization_function(Ps, mask_points_all, number_of_normalization_points, number_of_cameras, masks_all):
    P_0 = Ps[0]
    Fs = get_fundamental_matrices(P_0, Ps)
    P_0_center = np.linalg.svd(P_0)[-1][-1, :]
    P_0_center = P_0_center / P_0_center[3]

    # Use image 0 as a references
    xs = mask_points_all[0][0, :]
    ys = mask_points_all[0][1, :]

    counter = 0
    all_Xs = []

    # sample a subset of 2D points from camera 0
    indss = np.random.choice(xs.shape[0], (number_of_normalization_points,))
    pbar = tqdm(total=len(indss))
    for i in indss:
        curx = xs[i]
        cury = ys[i]
        # for each point, check its min/max depth in all other cameras.
        # If there is an intersection of relevant depth keep the point
        observerved_in_all = True
        max_d_all = 1e10
        min_d_all = 1e-10
        
        for j in range(1, number_of_cameras, 5):
            min_d, max_d = get_min_max_d(curx, cury, Ps[j], mask_points_all[j], P_0, Fs[j], j)
            if abs(min_d) < 0.00001:
                observerved_in_all = False
                break
            max_d_all = np.min(np.array([max_d_all, max_d]))
            min_d_all = np.max(np.array([min_d_all, min_d]))
            if max_d_all < min_d_all + 1e-2:
                observerved_in_all = False
                break

        if observerved_in_all:
            direction = np.linalg.inv(P_0[:3, :3]) @ np.array([curx, cury, 1.0])
            all_Xs.append(P_0_center[:3] + direction * min_d_all)
            all_Xs.append(P_0_center[:3] + direction * max_d_all)
            counter = counter + 1
            
        pbar.desc = str(counter)
        pbar.update(n=1)
        pbar.refresh()

    print("Number of points: %d" % counter)
    centroid = np.array(all_Xs).mean(axis=0)
    # mean_norm=np.linalg.norm(np.array(allXs)-centroid,axis=1).mean()
    scale = np.array(all_Xs).std()

    # OPTIONAL: refine the visual hull
    centroid, scale, all_Xs = refine_visual_hull(masks_all, Ps, scale, centroid)

    normalization = np.eye(4).astype(np.float32)

    normalization[0, 3] = centroid[0]
    normalization[1, 3] = centroid[1]
    normalization[2, 3] = centroid[2]

    normalization[0, 0] = scale
    normalization[1, 1] = scale
    normalization[2, 2] = scale
    return normalization, all_Xs


def get_normalization(source_dir, use_linear_init=False):
    print('Preprocessing', source_dir)

    if use_linear_init:
        # Since there is noise in the cameras, some of them will not apear in all the cameras, so we need more points
        number_of_normalization_points = 10000
        cameras_filename = "cameras_linear_init"
    else:
        number_of_normalization_points = 10000
        cameras_filename = "cameras"

    masks_dir = '{0}/mask'.format(source_dir)
    cameras = np.load('{0}/{1}.npz'.format(source_dir, cameras_filename))

    mask_points_all, masks_all = get_all_mask_points(masks_dir)
    number_of_cameras = len(masks_all)
    Ps = get_Ps(cameras, number_of_cameras)

    normalization, all_Xs = get_normalization_function(Ps, mask_points_all, number_of_normalization_points, number_of_cameras, masks_all)

    cameras_new = {}
    for i in range(number_of_cameras):
        cameras_new['scale_mat_%d' % i] = normalization
        cameras_new['world_mat_%d' % i] = np.concatenate((Ps[i], np.array([[0, 0, 0, 1.0]])), axis=0).astype(np.float32)

    np.savez('{0}/{1}_sphere.npz'.format(source_dir, cameras_filename), **cameras_new)

    print(normalization)
    print('--------------------------------------------------------')

    if False:  # for debugging
        for i in range(number_of_cameras):
            plt.figure()

            plt.imshow(mpimg.imread('%s/%03d.png' % (masks_path, i)))
            xy = (Ps[i, :2, :] @ (np.concatenate((np.array(all_Xs), np.ones((len(all_Xs), 1))), axis=1).T)) / (
                Ps[i, 2, :] @ (np.concatenate((np.array(all_Xs), np.ones((len(all_Xs), 1))), axis=1).T))

            plt.plot(xy[0, :], xy[1, :], '*')
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='', help='data source folder for preprocess')
    parser.add_argument('--dtu', default=False, action="store_true", help='If set, apply preprocess to all DTU scenes.')
    parser.add_argument('--use_linear_init', default=False, action="store_true", help='If set, preprocess for linear init cameras.')

    opt = parser.parse_args()

    if opt.dtu:
        source_dir = '../data/DTU'
        scene_dirs = sorted(glob(os.path.join(source_dir, "scan*")))
        for scene_dir in scene_dirs:
            get_normalization(scene_dir, opt.use_linear_init)
    else:
        get_normalization(opt.source_dir, opt.use_linear_init)

    print('Done!')
