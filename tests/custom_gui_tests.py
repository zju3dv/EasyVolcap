from easyvolcap.utils.console_utils import *
from easyvolcap.runners.custom_viewer import Viewer


@catch_throw
def main():
    viewer = Viewer()
    viewer.run()

if __name__ == '__main__':
    main()