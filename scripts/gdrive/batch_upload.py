"""
Upload files in batch using google drive upload
"""
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution


@catch_throw
def main():
    args = dotdict(
        input='/nas/home/xuzhen/datasets/vhulls.tar.gz.split',
        only=[]
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    # split -b 2048M  ../enerf_outdoor.tar.gz

    pattern = 'gdrive_linux_amd64 upload "{file}" --recursive'

    def upload_folder(f, p=pattern):
        run(p.format(file=f'{args.input}/{f}'))

    def upload(f, p=pattern):
        run(p.format(file=f))

    # Will have finished
    if isdir(args.input):
        files = os.listdir(args.input)
        if len(args.only): files = [f for f in files if f in args.only]
        parallel_execution(files, action=upload_folder)
    else:
        upload(args.input)


if __name__ == '__main__':
    main()
