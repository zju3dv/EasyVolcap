"""
Copy the first frame to a bkgd folder for colmap and other processing
"""

from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict(
        data_root='data/selfcap/0512_bike',

        bkgd_dir='bkgd',
        colmap_dir='colmap',
        images_dir='images',

        image='000000.png',
    )

    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    bkgd_dir = join(args.data_root, args.bkgd_dir)
    bkgd_images_dir = join(args.data_root, args.bkgd_dir, args.images_dir)
    bkgd_colmap_images_dir = join(args.data_root, args.bkgd_dir, args.colmap_dir, args.images_dir)
    images_dir = join(args.data_root, args.images_dir)
    os.makedirs(bkgd_dir, exist_ok=True)
    os.makedirs(bkgd_images_dir, exist_ok=True)
    os.makedirs(bkgd_colmap_images_dir, exist_ok=True)
    for cam in sorted(os.listdir(images_dir)):
        img = join(images_dir, cam, args.image)
        run(f'cp {img} {join(bkgd_dir, cam+splitext(args.image)[-1])}')

        os.makedirs(join(bkgd_images_dir, cam), exist_ok=True)
        run(f'cp {img} {join(bkgd_images_dir, cam, args.image)}')

        os.makedirs(join(bkgd_colmap_images_dir, cam), exist_ok=True)
        run(f'cp {img} {join(bkgd_colmap_images_dir, cam+splitext(args.image)[-1])}')


if __name__ == '__main__':
    main()
