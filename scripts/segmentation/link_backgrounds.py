"""
This script will link the flattened background images, 00.jpg 01.jpg etc. to their respective folders -> images/00/000000.jpg
So that we maintain a managable data structure for background reconstruction and camera parameter optimization
"""

from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict(
        data_root='data/selfcap/0330_01',
        bkgd_dir='colmap/colmap_dense/0/images',
        images_dir='colmap/images',
        image_base='000000',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    bkgd_dir = join(args.data_root, args.bkgd_dir)
    images_dir = join(args.data_root, args.images_dir)

    for bg in sorted(os.listdir(bkgd_dir)):
        if isdir(join(bkgd_dir, bg)): continue
        base, ext = splitext(bg)
        cam_dir = join(images_dir, base)
        img_name = args.image_base + ext
        os.makedirs(cam_dir, exist_ok=True)
        os.symlink(relpath(join(bkgd_dir, bg), cam_dir), join(cam_dir, img_name))

    log(yellow(f'Linked background images to {blue(bkgd_dir)}'))


if __name__ == '__main__':
    main()
