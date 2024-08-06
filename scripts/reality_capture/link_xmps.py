"""
Link the converted xmp files to their respective folders in the colmap directory.
"""

from easyvolcap.utils.console_utils import *
from glob import glob


@catch_throw
def main():
    args = dotdict(
        data_root='data/bullet/final',
        xmps_dir='xmps',
        colmap_dir='colmap',
        images_dir='images',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    xmps = glob(join(args.data_root, args.xmps_dir, '*.xmp'))
    imgs = os.listdir(join(args.data_root, args.colmap_dir))
    pbar = tqdm(total=len(xmps) * len(imgs))

    for img in imgs:
        for xmp in xmps:
            tar = join(args.data_root, args.colmap_dir, img, args.images_dir, split(xmp)[-1])
            src = xmp
            if exists(tar): os.remove(tar)
            os.symlink(relpath(src, dirname(tar)), tar)
            pbar.update()


if __name__ == '__main__':
    main()
