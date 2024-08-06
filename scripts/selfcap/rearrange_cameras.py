"""
Rearrange the extri.yml and intri.yml to start with 0000
And also fuse the t offset into the camera parameters
"""
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera, write_camera


@catch_throw
def main():
    args = dotdict(
        input='data/selfcap/0330_01/distorted',
        output='data/selfcap/0330_01/distorted',
        format='04d',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    output = dotdict()
    cameras = read_camera(args.input)
    names = sorted(cameras, key=lambda x: int(x))
    args.format = f'{{name:{args.format}}}'
    for i, name in enumerate(names):
        output[args.format.format(name=i)] = cameras[name]

    write_camera(output, args.output)


if __name__ == '__main__':
    main()
