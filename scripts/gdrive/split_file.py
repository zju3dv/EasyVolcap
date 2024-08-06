
"""
Split files into chunks for goolge drive upload
"""
from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict(
        input=dotdict(default='/nas/home/xuzhen/datasets/vhulls.tar.gz'),
        chunk=dotdict(default=1024)  # 2GB small files
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    input = args.input
    output = args.input + '.split'
    prefix = basename(args.input) + '.'

    run(f'mkdir -p {output}')
    run(f'split -d -b {args.chunk}M {input} {output}/{prefix}')  # prepare for uploading
    # for f in os.listdir(output):
    # run(f'mv {output}/{f} {output}/{prefix}{f}')


if __name__ == '__main__':
    main()
