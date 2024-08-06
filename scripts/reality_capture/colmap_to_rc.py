"""
Converts COLMAP format camera parameters to RealityCapture xmps
"""
import shutil
from glob import glob
from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict(
        data_root='data/bullet/final',
        image_ext='.jpg',
        images_dir='bkgd',
        colmap_dir='bkgd/colmap/colmap_text',
        reality_capture='C:/"Program Files"/"Capturing Reality"/RealityCapture/RealityCapture.exe',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    args.data_root = abspath(args.data_root)
    args.colmap_dir = join(args.data_root, args.colmap_dir)

    # Gotta first link all images to the text folder for RealityCapture to recognize
    images = glob(join(args.data_root, args.images_dir, '*' + args.image_ext))
    for src in tqdm(images):
        tar = join(args.colmap_dir, basename(src))
        # os.symlink(relpath(src, dirname(tar)), tar)
        shutil.copy(src, tar)

    # Run RealityCapture to load the COLMAP format camera parameters and export them as XMPs
    # FIXME: Not loading correctly
    run(f'{args.reality_capture} -load {args.colmap_dir}/images.txt -exportXMP -set "xmpCamera=3" -quit')


if __name__ == '__main__':
    main()
