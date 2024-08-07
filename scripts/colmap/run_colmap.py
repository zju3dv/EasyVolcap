"""
convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place
"""

import os
import cv2
import shutil
import argparse
import numpy as np
from glob import glob
from os.path import join
from pathlib import Path

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import load_mask, save_mask
from easyvolcap.utils.colmap_utils import correct_colmap_scale


def parse_args():
    """
    ffmpeg -i videos/00/IMG_6476.MOV -vf "fps=4,scale=1920:1080" -q:v 1 -qmin 1 -start_number 0 images/00/%06d.jpg
    ffmpeg -i videos/00/IMG_7348.MOV -vf "fps=30" -q:v 1 -qmin 1 -start_number 0 images/00/%06d.jpg
    """

    args = dotdict(
        colmap_db=dotdict(default="colmap/colmap.db", help="colmap database filename"),
        colmap_matcher=dotdict(default="sequential", choices=["exhaustive", "sequential", "spatial", "transitive", "vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images"),
        colmap_camera_model=dotdict(default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"], help="camera model"),
        colmap_camera_params=dotdict(default="", help="intrinsic parameters, depending on the chosen model.  Format: fx, fy, cx, cy, dist"),
        no_colmap_single_camera=dotdict(action='store_false', dest='colmap_single_camera'),

        data_root=dotdict(default='data/zju/ip412'),
        mask=dotdict(default="mask/00", help="input path to the mask"),
        images=dotdict(default="images/00", help="input path to the images"),
        bkgd_mask=dotdict(default="bkgd_mask/00", help="input path to the mask"),
        bkgd_mask_dilation=dotdict(default=20, type=int),

        ba_global_max_num_iterations=dotdict(default=20, type=int),
        ba_global_max_refinements=dotdict(default=3, type=int),
        max_num_iterations=dotdict(default=20, type=int),

        text=dotdict(default="text", help="input path to the colmap text files (set automatically if run_colmap is used)"),
        ply=dotdict(default="ply", help="output sparse reconstruction results"),
        vocab_path=dotdict(default="", help="vocabulary tree path"),

        use_glomap=dotdict(action='store_true'),
        use_gpu=dotdict(action='store_true'),

    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    args.mask = join(args.data_root, args.mask)
    args.images = join(args.data_root, args.images)
    args.colmap_db = join(args.data_root, args.colmap_db)
    args.bkgd_mask = join(args.data_root, args.bkgd_mask)

    return args


def bkgd_one_mask(mask_path, output_path, dilation=20):
    msk = load_mask(mask_path)
    kernel = np.ones((dilation, dilation), np.uint8)
    msk = cv2.dilate(msk.astype(np.uint8), kernel)[..., None] > 0  # why is the last channel gone?
    msk = ~msk
    save_mask(output_path, msk)


def bkgd_mask(args):
    if os.path.exists(args.bkgd_mask):
        log('skipping mask inversion')
        return
    elif not os.path.exists(args.mask):
        log(yellow(f'original mask directory: {args.mask} does not exist, skipping mask inversion'))
        return
    log(f'performing mask inversion from {args.mask} to {args.bkgd_mask}')
    img_paths = glob(os.path.join(args.images, '*'))  # 000000.jpg
    mask_paths = [f.replace(args.images, args.mask) for f in img_paths]  # 000000.jpg, yep...
    if not os.path.exists(mask_paths[0]):
        mask_paths = [f.replace(args.images, args.mask).replace('.jpg', '.png') for f in img_paths]  # 000000.png
    output_paths = [f.replace(args.images, args.bkgd_mask) + ".png" for f in img_paths]  # 000000.jpg.png
    parallel_execution(mask_paths, output_paths, dilation=args.bkgd_mask_dilation, action=bkgd_one_mask, print_progress=True)


def run_colmap(args):
    # prepare output dirs
    db = args.colmap_db
    run(f'mkdir -p {os.path.dirname(db)}')
    images = "\"" + args.images + "\""
    db_noext = str(Path(db).with_suffix(""))
    if args.text == "text":
        args.text = db_noext + "_text"
    if args.ply == "ply":
        args.ply = db_noext + "_ply"
    if args.ply == "surfs" or args.ply == 'vhulls' or args.ply == 'sparse':
        args.ply = join(args.data_root, args.ply)
    ply = args.ply
    text = args.text
    sparse = db_noext + "_sparse"
    dense = sparse.replace('sparse', 'dense')

    log(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
    if os.path.isdir(sparse) or os.path.isdir(text) or os.path.isfile(db):  # be caring
        print(yellow(f"warning! '{red(sparse)}' '{red(text)}' and '{red(db)}' will be deleted/replaced. continue? (Y/n) "), end='')
        if (input().lower().strip() + "y")[:1] != "y":
            sys.exit(1)
        try:
            os.remove(db)
            shutil.rmtree(sparse, ignore_errors=True)
            shutil.rmtree(text, ignore_errors=True)
            shutil.rmtree(ply, ignore_errors=True)
        except:
            pass  # whatever

    # feature extraction: TODO: use gpu
    cmd = f"colmap feature_extractor \
--ImageReader.camera_model {args.colmap_camera_model} \
--ImageReader.camera_params \"{args.colmap_camera_params}\" \
--SiftExtraction.estimate_affine_shape=true \
--SiftExtraction.domain_size_pooling=true \
--ImageReader.single_camera {1 if args.colmap_single_camera else 0} \
--database_path {db} \
--image_path {images} \
"

    if args.use_gpu:
        cmd += "--SiftExtraction.use_gpu 1 "
    else:
        cmd += "--SiftExtraction.use_gpu 0 "

    if os.path.exists(args.bkgd_mask):
        cmd = f'{cmd} --ImageReader.mask_path \"{args.bkgd_mask}\"'
    else:
        log(yellow(f'{args.bkgd_mask} does not exist, colmap will not use mask'))
    run(cmd)

    # feature matching
    match_cmd = f"colmap {args.colmap_matcher}_matcher \
--SiftMatching.guided_matching=true \
--database_path {db} \
"

    if args.use_gpu:
        match_cmd += "--SiftMatching.use_gpu 1 "
    else:
        match_cmd += "--SiftMatching.use_gpu 0 "

    if args.vocab_path:
        match_cmd += f"--VocabTreeMatching.vocab_tree_path {args.vocab_path} "
    run(match_cmd)

    # mapping (sparse reconstruction) -> camera poses
    os.makedirs(sparse, exist_ok=True)
    tolerance = {
        '--Mapper.ba_global_function_tolerance': 1e-6,
        '--Mapper.ba_global_max_num_iterations': args.ba_global_max_num_iterations,
        '--Mapper.ba_global_max_refinements': args.ba_global_max_refinements,
    }
    tolerance_string = ' '.join([f'{k} {v}' for k, v in tolerance.items()])
    log('running colmap mapper, this could take forever, go to sleep plz')
    log(yellow(f'using mapper tolerance parameters: {tolerance_string}'))

    # backend = 'glomap' if args.use_glomap else 'colmap'
    # run(f"{backend} mapper {tolerance_string} --database_path {db} --image_path {images} --output_path {sparse}")
    if args.use_glomap:
        run(f"glomap mapper --database_path {db} --image_path {images} --output_path {sparse}")
    else:
        run(f"colmap mapper {tolerance_string} --database_path {db} --image_path {images} --output_path {sparse}")

    # global bundle adjustment for principal point
    tolerance = {
        '--BundleAdjustment.refine_principal_point': 1,
        '--BundleAdjustment.max_num_iterations': args.max_num_iterations,
        '--BundleAdjustment.function_tolerance': 1e-6,
        '--BundleAdjustment.gradient_tolerance': 1e-10,
        '--BundleAdjustment.parameter_tolerance': 1e-8,
    }
    tolerance_string = ' '.join([f'{k} {v}' for k, v in tolerance.items()])
    log('running colmap bundle_adjuster, this could take a smaller forever, grab a coffee plz')
    log(yellow(f'using bundle_adjuster tolerance parameters: {tolerance_string}'))
    run(f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 {tolerance_string}")

    run(f"colmap model_orientation_aligner --image_path {images} --input_path {sparse}/0 --output_path {sparse}/0")  # TODO: Repeated undistortion here

    os.makedirs(f'{dense}/0', exist_ok=True)
    run(f"colmap image_undistorter --image_path {images} --input_path {sparse}/0 --output_path {dense}/0")

    os.makedirs(text, exist_ok=True)
    run(f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")
    os.makedirs(ply, exist_ok=True)
    run(f"colmap model_converter --input_path {sparse}/0 --output_path {ply}/000000.ply --output_type PLY")
    log(green(f'colmap finished, check the sparse reconstruction results under: {sparse}, {text} and {ply}'))
    # correct_colmap_scale(sparse, text)


def main():
    args = parse_args()
    bkgd_mask(args)  # used for colmap
    run_colmap(args)  # used for literally everything


if __name__ == "__main__":
    main()


"""
[
6.804892 / 2.4,
6.900750 / 2.4,
6.896621 / 2.4,

10.840011 / (2.4**2 + 3.0**2)**0.5,
9.472128 / (2.4**2 + 2.4**2)**0.5,
]
scale = 0.3521977271755294

[
5.6114  / 2.4,
5.54979 / 2.4,
5.58637 / 2.4,

9.079804 / (2.4**2 + 3.0**2)**0.5,
7.873619 / (2.4**2 + 2.4**2)**0.5,
]
scale = 0.4287680803112663

[
5.708538 / 2.4,
5.599938 / 2.4,
5.646156 / 2.4,
7.807993 / (2.4**2 + 2.4**2)**0.5,
8.843090 / (2.4**2 + 3.0**2)**0.5,
]
scale = 0.42727045923959855

[
4.403869 / 2.4,
4.440443 / 2.4,
4.340834 / 2.4,
4.295539 / 2.4,
6.999857 / (2.4**2 + 3.0**2)**0.5,
6.197420 / (2.4**2 + 2.4**2)**0.5,
]
scale = 0.5488704592749379
"""
