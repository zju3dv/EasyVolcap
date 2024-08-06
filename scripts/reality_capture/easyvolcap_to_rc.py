"""
Convert EasyVolcap format camera parameters to RealityCapture
"""

import re
from easyvolcap.utils.easy_utils import read_camera, write_camera
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.rc_utils import ixt_to_rc, ext_to_rc
from functools import partial
# from copy import deepcopy

# import xml.etree.ElementTree as ET


@catch_throw
def main():
    args = dotdict(
        data_root='data/bullet/final',
        camears_dir='',
        xmps_dir='xmps',
        H=2160,
        W=3840,
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    cameras = read_camera(join(args.data_root, args.camears_dir))
    # tree = ET.parse(f'{dirname(__file__)}/reality_capture.xmp')
    with open(f'{dirname(__file__)}/reality_capture.xmp', 'r') as f:
        text = f.read()

    os.makedirs(join(args.data_root, args.xmps_dir), exist_ok=True)
    for name, camera in tqdm(cameras.items()):
        H, W, K, R, T, D = camera.H, camera.W, camera.K, camera.R, camera.T, camera.D
        H = H if H != -1 else args.H
        W = W if W != -1 else args.W
        log(R)
        # tree = deepcopy(tree)
        FocalLength35mm, Skew, AspectRatio, PrincipalPointU, PrincipalPointV, DistortionCoeficients = ixt_to_rc(K, D, H, W)
        FocalLength35mm, Skew, AspectRatio, PrincipalPointU, PrincipalPointV = format(FocalLength35mm, '.9f'), format(Skew, '.9f'), format(AspectRatio, '.9f'), format(PrincipalPointU, '.9f'), format(PrincipalPointV, '.9f')
        Rotation, Position = ext_to_rc(R, T)
        Rotation = ' '.join(map(lambda x: format(x, '.9f'), Rotation))
        Position = ' '.join(map(lambda x: format(x, '.9f'), Position))
        DistortionCoeficients = ' '.join(map(lambda x: format(x, '.9f'), DistortionCoeficients))

        text = re.sub(r'xcr:FocalLength35mm="(\d|\.|\s|-|\n)*"', f'xcr:FocalLength35mm="{FocalLength35mm}"', text)
        text = re.sub(r'xcr:Skew="(\d|\.|\s|-|\n)*"', f'xcr:Skew="{Skew}"', text)
        text = re.sub(r'xcr:AspectRatio="(\d|\.|\s|-|\n)*"', f'xcr:AspectRatio="{AspectRatio}"', text)
        text = re.sub(r'xcr:PrincipalPointU="(\d|\.|\s|-|\n)*"', f'xcr:PrincipalPointU="{PrincipalPointU}"', text)
        text = re.sub(r'xcr:PrincipalPointV="(\d|\.|\s|-|\n)*"', f'xcr:PrincipalPointV="{PrincipalPointV}"', text)
        text = re.sub(r'<xcr:Rotation>(\d|\.|\s|-|\n)*', f'<xcr:Rotation>{Rotation}', text)
        text = re.sub(r'<xcr:Position>(\d|\.|\s|-|\n)*', f'<xcr:Position>{Position}', text)
        text = re.sub(r'<xcr:DistortionCoeficients>(\d|\.|\s|-|\n)*', f'<xcr:DistortionCoeficients>{DistortionCoeficients}', text)

        # = ' '.join(map(str, Rotation))
        # = ' '.join(map(str, Position))
        # = ' '.join(map(str, DistortionCoeficients))

        # = str(FocalLength35mm)
        # = str(Skew)
        # = str(AspectRatio)
        # = str(PrincipalPointU)
        # = str(PrincipalPointV)

        file_path = join(args.data_root, args.xmps_dir, name + '.xmp')
        with open(file_path, 'w') as f:
            f.write(text)


if __name__ == '__main__':
    main()
