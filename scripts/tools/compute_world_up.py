"""
Compute the center world up vector for easier control in the viewer
"""
import torch
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.viewer_utils import Camera
from easyvolcap.utils.easy_utils import read_camera, write_camera
from easyvolcap.utils.data_utils import to_tensor, to_cpu, to_cuda
from easyvolcap.utils.math_utils import affine_inverse, normalize


@catch_throw
def main():
    args = dotdict(
        camera_dir='data/badminton/seq3'
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    cameras = read_camera(args.camera_dir)
    camera_names = sorted(cameras)
    cameras = dotdict({cam: cameras[cam] for cam in camera_names})
    batches = dotdict({cam: to_tensor(Camera().from_easymocap(cameras[cam]).to_batch()) for cam in camera_names})

    Rs = torch.stack([batches[cam].R for cam in batches])  # V, 3, 3
    Ts = torch.stack([batches[cam].T for cam in batches])  # V, 3, 1
    Ks = torch.stack([batches[cam].K for cam in batches])  # V, 3, 3

    # c2ws = affine_inverse(torch.cat([Rs, Ts], dim=-1))  # V, 3, 4
    world_up = normalize(Rs[:, 1, :].mean(dim=0), eps=0)
    log(f'Computed world up vector: {world_up}')


if __name__ == '__main__':
    main()
