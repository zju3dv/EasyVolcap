import math
import torch
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.math_utils import affine_inverse
from easyvolcap.utils.colmap_utils import qvec2rotmat, rotmat2qvec


def load_select_cameras(
    data_root,
    cameras_dir,
    intri_file='intri.yml',
    extri_file='extri.yml',
    view_sample=[0, None, 1],
    frame_sample=[0, None, 1],
    n_frames_total: int = 1,
):
    # Load camera related stuff like image list and intri, extri.
    # Determine whether it is a monocular dataset or multiview dataset based on the existence of root `extri.yml` or `intri.yml`
    # Multiview dataset loading, need to expand, will have redundant information
    if exists(join(data_root, intri_file)) and exists(join(data_root, extri_file)):
        cameras = read_camera(join(data_root, intri_file), join(data_root, extri_file))
        camera_names = np.asarray(sorted(list(cameras.keys())))  # NOTE: sorting camera names
        cameras = dotdict({k: [cameras[k] for i in range(n_frames_total)] for k in camera_names})

    # Monocular dataset loading, each camera has a separate folder
    elif exists(join(data_root, cameras_dir)):
        camera_names = np.asarray(sorted(os.listdir(join(data_root, cameras_dir))))  # NOTE: sorting here is very important!
        cameras = dotdict({
            k: [v[1] for v in sorted(
                read_camera(join(data_root, cameras_dir, k, intri_file),
                            join(data_root, cameras_dir, k, extri_file)).items()
            )] for k in camera_names
        })

    else:
        log(red('Could not find camera information in the dataset, check your dataset configuration'))
        log(red('If you want to render the model without loading anything from the dataset:'))
        log(red('Try appending val_dataloader_cfg.dataset_cfg.type=NoopDataset to your command or add the `configs/specs/turbom.yaml` to your `-c` parameter'))
        raise NotImplementedError(f'Could not find {{{intri_file},{extri_file}}} or {cameras_dir} directory in {data_root}, check your dataset configuration or use NoopDataset')

    # Expectation:
    # camera_names: a list containing all camera names
    # cameras: a mapping from camera names to a list of camera objects
    # (every element in list is an actual camera for that particular view and frame)
    # NOTE: ALWAYS, ALWAYS, SORT CAMERA NAMES.
    Hs = torch.as_tensor([[cam.H for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # V, F
    Ws = torch.as_tensor([[cam.W for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # V, F
    Ks = torch.as_tensor([[cam.K for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # V, F, 3, 3
    Rs = torch.as_tensor([[cam.R for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # V, F, 3, 3
    Ts = torch.as_tensor([[cam.T for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # V, F, 3, 1
    Ds = torch.as_tensor([[cam.D for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # V, F, 1, 5
    Cs = -Rs.mT @ Ts  # V, F, 3, 1
    w2cs = torch.cat([Rs, Ts], dim=-1)  # V, F, 3, 4
    c2ws = affine_inverse(w2cs)  # V, F, 3, 4

    # Minus the average translation, this is the camera center in world space
    c2ws[..., :3, 3] = c2ws[..., :3, 3] - c2ws[..., :3, 3].reshape(-1, 3).mean(dim=0)[None, None]
    w2cs = affine_inverse(c2ws)  # V, F, 3, 4
    Rs = w2cs[..., :3, :3]  # V, F, 3, 3
    Ts = w2cs[..., :3, 3:]  # V, F, 3, 1

    # Only retrain needed
    # Perform view selection first
    view_inds = torch.arange(Ks.shape[0])
    if len(view_sample) != 3: view_inds = view_inds[view_sample]  # this is a list of indices
    else: view_inds = view_inds[view_sample[0]:view_sample[1]:view_sample[2]]  # begin, start, end
    if len(view_inds) == 1: view_inds = [view_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

    # Perform frame selection next
    frame_inds = torch.arange(Ks.shape[1])
    if len(frame_sample) != 3: frame_inds = frame_inds[frame_sample]
    else: frame_inds = frame_inds[frame_sample[0]:frame_sample[1]:frame_sample[2]]
    if len(frame_inds) == 1: frame_inds = [frame_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

    # NOTE: if view_inds == [0,] in monocular dataset or whatever case, type(`camera_names[view_inds]`) == str, not a list of str
    camera_names = np.asarray([camera_names[view] for view in view_inds])  # this is what the b, e, s means
    cameras = dotdict({k: [cameras[k][int(i)] for i in frame_inds] for k in camera_names})  # reloading
    Hs = Hs[view_inds][:, frame_inds]
    Ws = Ws[view_inds][:, frame_inds]
    Ks = Ks[view_inds][:, frame_inds]
    Rs = Rs[view_inds][:, frame_inds]
    Ts = Ts[view_inds][:, frame_inds]
    Ds = Ds[view_inds][:, frame_inds]
    Cs = Cs[view_inds][:, frame_inds]
    w2cs = w2cs[view_inds][:, frame_inds]
    c2ws = c2ws[view_inds][:, frame_inds]

    return camera_names, cameras, Hs, Ws, Ks, Rs, Ts, Ds, Cs, w2cs, c2ws


def fov2focal(
    fov: float,
    H: int = 1080,
    W: int = 1920,
    type: str = 'vfov',
    is_deg: bool = True
):
    # Convert fov to radian
    if is_deg: fov = math.radians(fov)
    # Compute focal length
    if type == 'hfov': focal = W / (2 * math.tan(fov / 2))
    elif type == 'vfov': focal = H / (2 * math.tan(fov / 2))
    elif type == 'dfov': focal = math.sqrt(H**2 + W**2) / (2 * math.tan(fov / 2))
    return focal


def reshade2opencv(c2w, type='2077'):
    """ Convert reshade camera to opencv camera """

    if '2077' in type or 'rdr2' in type:
        # Cyberpunk 2077 camera
        c2w[[1, 2], :] = c2w[[2, 1], :]
        c2w[:3, [1, 2]] = c2w[:3, [2, 1]]

        c2w[1, :] = -c2w[1, :]
        c2w[:, 1] = -c2w[:, 1]
        c2w[[0, 2], :] = -c2w[[0, 2], :]

    elif 'wukong' in type:
        # Black Myth: Wukong camera
        c2w[[0, 2], :] = c2w[[2, 0], :]
        c2w[:3, [0, 2]] = c2w[:3, [2, 0]]

        c2w[:, [0, 1]] = c2w[:, [1, 0]]
        c2w[:, 1] = -c2w[:, 1]

    else:
        raise ValueError(f'Unknown camera type: {type}')

    return c2w


def load_igcs_pose_v0(
    df: pd.DataFrame,
    H: int = 1080,
    W: int = 1920,
    fov: float = None,
    focal: float = None,
    fov_type: str = 'vfov',
    type: str = '2077',
    scale: float = 1.0,
):
    # Load the camera poses from the dataframe
    cameras = dotdict()
    Ks, c2ws = [], []

    for i, row in df.iterrows():
        # Compute the intrinsic matrix
        if focal is not None: focal = focal
        elif fov is not None: focal = fov2focal(fov, H, W, fov_type, is_deg=True)
        else: focal = fov2focal(row['fov'], H, W, fov_type, is_deg=True)
        K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])
        Ks.append(K)

        # Compute the extrinsic matrix
        c2w = np.eye(4)
        c2w[:3, :3] = qvec2rotmat(
            np.array([row['qw'], row['qx'], row['qy'], row['qz']])
        )
        c2w[:3, 3] = np.array([row['x'], row['y'], row['z']]) * scale
        c2w = reshade2opencv(c2w, type=type)
        c2ws.append(c2w)

    # Stack the camera parameters
    Ks, c2ws = np.stack(Ks), np.stack(c2ws)

    # Update the camera position with the average position
    c2ws[:, :3, 3] = c2ws[:, :3, 3] - np.mean(c2ws[:, :3, 3], axis=0, keepdims=True)  # (N, 3)
    w2cs = np.linalg.inv(c2ws)  # (N, 4, 4)
    Rs, Ts = w2cs[:, :3, :3], w2cs[:, :3, 3]  # (N, 3, 3), (N, 3)

    # Update the camera poses
    for i in range(len(Ks)):
        cameras[f'{i:06d}'] = dotdict(
            H=H, W=W, K=Ks[i], R=Rs[i], T=Ts[i]
        )

    return cameras, Ks, Rs, Ts, w2cs, c2ws


def check_all_same_row(
    df: pd.DataFrame,
    row: pd.Series,
    check_keys: List[str] = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'],
):
    v0 = np.array([row[key] for key in check_keys])
    v1 = np.stack([df[key] for key in check_keys], axis=-1)
    return np.isclose(v0, v1).all(-1).all()


def load_igcs_pose_v1(
    df: pd.DataFrame,
    H: int = 1080,
    W: int = 1920,
    fov: float = None,
    focal: float = None,
    fov_type: str = 'vfov',
    type: str = '2077',
    scale: float = 1.0,
    delta: int = 0,
    check_keys: List[str] = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'],
):
    # Load the camera poses from the dataframe
    cameras = dotdict()
    Ks, c2ws = [], []

    # Iterate over the key frames
    for _, r in df[df['frameIndex'] >= 0].iterrows():
        # Filter rows with the same game index
        gidx = r['gameFrameIndex'] + delta
        df_gidx = df[df['gameFrameIndex'] == gidx]
        df_gidx = df_gidx[df_gidx['frameIndex'] < 0]

        # The first row is the camera parameters
        row = df_gidx.iloc[0].copy()
        idx = int(r['frameIndex'])

        # Check if the camera parameters are the same
        if len(df_gidx) > 0:
            if not check_all_same_row(
                df_gidx[:-1], row, check_keys=check_keys
            ):
                log(f'Camera parameters are not the same for game index {gidx}')
                raise ValueError('Please check with xiaoyang')
        else:
            log(f'No camera parameters found for game index {gidx}')
            raise ValueError('Please check with xiaoyang')

        # Compute the intrinsic matrix
        if focal is not None: focal = focal
        elif fov is not None: focal = fov2focal(fov, H, W, fov_type, is_deg=True)
        else: focal = fov2focal(row['fov'], H, W, fov_type, is_deg=True)
        K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])
        Ks.append(K)

        # Compute the extrinsic matrix
        c2w = np.eye(4)
        c2w[:3, :3] = qvec2rotmat(
            np.array([row['qw'], row['qx'], row['qy'], row['qz']])
        )
        c2w[:3, 3] = np.array([row['x'], row['y'], row['z']]) * scale
        c2w = reshade2opencv(c2w, type=type)
        c2ws.append(c2w)

    # Stack the camera parameters
    Ks, c2ws = np.stack(Ks), np.stack(c2ws)

    # Update the camera position with the average position
    c2ws[:, :3, 3] = c2ws[:, :3, 3] - np.mean(c2ws[:, :3, 3], axis=0, keepdims=True)  # (N, 3)
    w2cs = np.linalg.inv(c2ws)  # (N, 4, 4)
    Rs, Ts = w2cs[:, :3, :3], w2cs[:, :3, 3]  # (N, 3, 3), (N, 3)

    # Update the camera poses
    for i in range(len(Ks)):
        cameras[f'{i:06d}'] = dotdict(
            H=H, W=W, K=Ks[i], R=Rs[i], T=Ts[i]
        )

    return cameras, Ks, Rs, Ts, w2cs, c2ws
