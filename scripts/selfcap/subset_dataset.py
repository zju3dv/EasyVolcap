"""
This script will try to create a subset of the dataset for running LongVolcap
"""
from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict(
        input_dir='/mnt/data/home/xuzhen/datasets/selfcap/0525_corgi',
        output_dir='/mnt/data/home/xuzhen/datasets/selfcap/0525_corgi_s5200_e8700',
        image_dirs=['images', 'masks', 'depths', 'normals'],
        camera_dirs=['cameras'],
        share_dirs=['colmap', 'calib', 'bkgd', 'intri.yml', 'extri.yml'],
        frame_sample=[5200, 8700, 1],
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    os.makedirs(args.output_dir, exist_ok=True)
    b, e, s = args.frame_sample

    for image_dir in args.image_dirs:
        if exists(join(args.input_dir, image_dir)):
            os.makedirs(join(args.output_dir, image_dir), exist_ok=True)
            for cam in sorted(os.listdir(join(args.input_dir, image_dir))):
                os.makedirs(join(args.output_dir, image_dir, cam), exist_ok=True)
                for i, file in enumerate(tqdm(sorted(os.listdir(join(args.input_dir, image_dir, cam)))[b:e:s], desc=join(image_dir, cam))):
                    src = join(args.input_dir, image_dir, cam, file)
                    tar = join(args.output_dir, image_dir, cam, f'{i:06d}{splitext(file)[-1]}')
                    os.symlink(relpath(src, dirname(tar)), tar)

    for camera_dir in args.camera_dirs:
        if exists(join(args.input_dir, camera_dir)):
            os.makedirs(join(args.output_dir, camera_dir), exist_ok=True)
            for i, file in enumerate(tqdm(sorted(os.listdir(join(args.input_dir, camera_dir)))[b:e:s], desc=camera_dir)):
                src = join(args.input_dir, camera_dir, file)
                tar = join(args.output_dir, camera_dir, f'{i:06d}{splitext(file)[-1]}')
                os.symlink(relpath(src, dirname(tar)), tar)

    for share_dir in args.share_dirs:
        if exists(join(args.input_dir, share_dir)):
            src = join(args.input_dir, share_dir)
            tar = join(args.output_dir, share_dir)
            os.symlink(relpath(src, dirname(tar)), tar)


if __name__ == '__main__':
    main()
