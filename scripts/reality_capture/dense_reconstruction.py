"""
Perform dense reconstruction on the background images using RealityCapture.
"""
import shutil
from glob import glob
from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict(
        data_root='data/bullet/final',
        colmap_dir='colmap',
        images_dir='images',
        xmps_dir='xmps',
        pcds_dir='dense_xyzs_rc',
        reality_capture='C:/"Program Files"/"Capturing Reality"/RealityCapture/RealityCapture.exe',
        frame_sample=[0, 30000, 1000],
        recopy_xmps=False,
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    args.data_root = abspath(args.data_root)
    args.xmps_dir = join(args.data_root, args.xmps_dir)
    args.pcds_dir = join(args.data_root, args.pcds_dir)

    os.makedirs(args.pcds_dir, exist_ok=True)
    frames = [f'{i:06d}' for i in range(*args.frame_sample)]
    for frame in frames:

        images_dir = join(args.data_root, args.colmap_dir, frame, args.images_dir)
        xmps = glob(join(args.xmps_dir, '*.xmp'))
        for xmp in tqdm(xmps):
            tar = join(images_dir, basename(xmp))
            if exists(tar):
                if args.recopy_xmps:
                    os.remove(tar)
            if not exists(tar):
                shutil.copyfile(xmp, tar)
                # os.symlink(relpath(xmp, dirname(tar)), tar) # windows symlinks will not work on linux, so we copy, since this is a dense reconstruction

        # Run RealityCapture to perform dense reconstruction
        run(f'{args.reality_capture} -set "appAutoSaveMode=false" -set "appAutoSaveCliHandling=delete" -set "sfmEnableCameraPrior=true" -addFolder {images_dir} -align -setReconstructionRegionAuto -calculateNormalModel -selectMarginalTriangles -removeSelectedTriangles -renameSelectedModel {images_dir} -calculateTexture -exportModel {images_dir} {join(args.pcds_dir, frame + ".xyz")} -quit')


if __name__ == '__main__':
    main()
