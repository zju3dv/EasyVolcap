"""
Given two sets for camera parameters
Compute their optimal transform and transform one of the sets to another
"""

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from torch.optim import Adam
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera, write_camera
from easyvolcap.utils.data_utils import to_tensor, to_cuda, to_cpu, to_numpy, to_x
from easyvolcap.utils.math_utils import affine_inverse, affine_padding
from easyvolcap.utils.loss_utils import mse
from easyvolcap.utils.blend_utils import batch_rodrigues


@catch_throw
def main():
    args = dotdict(
        src='data/selfcap/0512_bike/cameras_037300',
        tar='data/selfcap/0512_bike',
        output='data/selfcap/0512_bike/aligned',
        lr=1e-1,
        iter=1000,
        init_idx=9,
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    src_cams = read_camera(args.src)
    tar_cams = read_camera(args.tar)
    names = sorted(src_cams)
    src_cams_list = to_x(to_tensor([src_cams[key] for key in names]), torch.float)
    tar_cams_list = to_x(to_tensor([tar_cams[key] for key in names]), torch.float)

    Rs = to_cuda(torch.stack([cam.R for cam in src_cams_list]))
    Ts = to_cuda(torch.stack([cam.T for cam in src_cams_list]))
    As = affine_padding(torch.cat([Rs, Ts], dim=-1))

    Rs_gt = to_cuda(torch.stack([cam.R for cam in tar_cams_list]))
    Ts_gt = to_cuda(torch.stack([cam.T for cam in tar_cams_list]))
    As_gt = affine_padding(torch.cat([Rs_gt, Ts_gt], dim=-1))

    w2c_src = affine_padding(torch.cat([Rs[args.init_idx], Ts[args.init_idx]], dim=-1))
    w2c_tar = affine_padding(torch.cat([Rs_gt[args.init_idx], Ts_gt[args.init_idx]], dim=-1))
    trans = affine_inverse(w2c_tar) @ w2c_src

    r = matrix_to_axis_angle(trans[:3, :3])
    t = trans[:3, 3:]
    s = torch.as_tensor(1, dtype=torch.float).to(r.device, non_blocking=True)
    r.requires_grad_()
    t.requires_grad_()
    s.requires_grad_()
    optimizer = Adam([r, t, s], lr=args.lr)

    pbar = tqdm(range(args.iter))
    for i in pbar:
        A = affine_padding(torch.cat([batch_rodrigues(r), t], dim=-1))
        As_pred = affine_inverse(s * A @ affine_inverse(As))

        loss = mse(As_pred, As_gt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        pbar.update()
        pbar.set_description(str(loss.item()))

    cameras = dotdict()
    for i in range(len(names)):
        R, T = As_pred[i, :3, :3], As_pred[i, :3, 3:]
        cameras[names[i]] = dotdict(R=R, T=T, K=src_cams[names[i]].K, D=src_cams[names[i]].D, H=src_cams[names[i]].H, W=src_cams[names[i]].W)
    cameras = to_numpy(cameras)

    log(yellow(f'Aligned cameras saved to: {blue(args.output)}'))
    write_camera(cameras, args.output)


if __name__ == '__main__':
    main()
