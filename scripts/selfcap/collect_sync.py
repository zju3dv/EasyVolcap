"""
Collect the t offset of the synchronization process, and feed this to temporal_forest_gaussian_splatting
"""
from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict(
        format='04d',
        data_root='data/selfcap/0525_corgi_s5200_e8700',
        frame_input='calib/flash_detect_results.json',
        subframe_input='calib/subframe_offset_result_by_cam_t_offset.json',
        fps=60,
        output='sync.json',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    subframe = json.load(open(join(args.data_root, args.subframe_input)))
    if exists(join(args.data_root, args.frame_input)):
        offset = json.load(open(join(args.data_root, args.frame_input)))['offset_dict']
        frame = dotdict()
        for name in offset:
            frame[f'{int(split(name)[-1]):02d}'] = offset[name] * 1 / args.fps
    else:
        frame = dotdict()
        for name in subframe:
            frame[name] = 0

    args.format = f'{{name:{args.format}}}'
    offset = dotdict()
    for i, name in enumerate(sorted(frame)):
        offset[args.format.format(name=i)] = frame[name] + subframe[name]

    json.dump(offset, open(join(args.data_root, args.output), mode='w'))
    log(yellow(f'Converted synchronization result saved to: {blue(join(args.data_root, args.output))}'))


if __name__ == '__main__':
    main()
