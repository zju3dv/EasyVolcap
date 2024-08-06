"""
Extracts multi-view images from multi-view videos using ffmpeg
"""

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution


def run_one(vid: str, images_dir: str, videos_dir: str, args: dotdict):
    out = join(images_dir, os.path.splitext(vid)[0])
    vid = join(videos_dir, vid)
    os.makedirs(out, exist_ok=True)
    cmd = args.cmd.format(video_path=vid, output_path=out)
    run(cmd)


@catch_throw
def main():
    args = dotdict(
        data_root='data/selfcap/0525_corgi',
        videos_dir='videos',
        images_dir='images',
        view_sample=[0, None, 1],
        ext='.jpg',
        image='06d',
        sequential=False,

        trim=False,
        ss='00:00:00',
        t='00:10:00',
        trim_vf='-ss {ss} -t {t}',

        x=0, y=0, w=1920, h=1080, width=1920, height=1080,
        crop=False,
        crop_vf='crop={w}:{h}:{x}:{y},scale={width}:{height}',

        hdr=False,
        hdr_vf='zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,',

        lut=False,
        lut_file='data/selfcap/0512_bike/bike.cube',
        lut_vf='lut3d={lut},',

        # cmd='/usr/bin/ffmpeg -hwaccel cuda -vcodec hevc_cuvid -i {{video_path}} -ss {ss} {vf}{lut}{hdr}{crop} -q:v 1 -qmin 1 -compression_level 100 -start_number 0 {{output_path}}/{image}{ext} -y'
        cmd='/usr/bin/ffmpeg -i {{video_path}} {trim} {vf}{lut}{hdr}{crop} -q:v 1 -qmin 1 -compression_level 100 -start_number 0 {{output_path}}/{image}{ext} -y'
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    args.cmd = args.cmd.format(trim=args.trim_vf.format(ss=args.ss, t=args.t) if args.trim else '',
                               lut=args.lut_vf.format(lut=args.lut_file) if args.lut else '',
                               image=f'%{args.image}', ext=args.ext,
                               hdr=args.hdr_vf if args.hdr else '',
                               crop=args.crop_vf.format(x=args.x, y=args.y, w=args.w, h=args.h, width=args.width, height=args.height) if args.crop else '',
                               vf='-vf' if args.hdr or args.crop or args.lut else '',
                               )
    log('Will use ffmpeg command:', yellow(args.cmd), 'for extracting videos')

    videos_dir = join(args.data_root, args.videos_dir)
    images_dir = join(args.data_root, args.images_dir)

    b, e, s = args.view_sample
    parallel_execution(sorted(os.listdir(videos_dir))[b:e:s], images_dir, videos_dir, args, action=run_one, sequential=args.sequential)


if __name__ == '__main__':
    main()
