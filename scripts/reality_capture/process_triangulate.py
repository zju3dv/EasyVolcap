import os
from os.path import join
import shutil
import time
# from tqdm import tqdm
from easyvolcap.utils.console_utils import *  # 用evc是因为想用evc的tqdm，如果没有evc环境，直接用原本的tqdm也行，就是print看起来有点不好看
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start_idx', default=0, type=int)
parser.add_argument('--end_idx', default=-1, type=int)
parser.add_argument('--remote_data_dir', default=r'Z:\root\mnt\selfcap\selfcap\0512_bike\colmap')
parser.add_argument('--temp_data_dir', default=r'temp')

args = parser.parse_args()
start_idx = args.start_idx
end_idx = args.end_idx

remote_data_dir = os.path.abspath(args.remote_data_dir)
temp_data_dir = os.path.abspath(args.temp_data_dir)
# remote_data_dir = r'D:\selfcap\0512_bike\colmap' # 必须是绝对路径
# data_dir = 'E:\\luoyunsheng\\code\\rc_scripts\\data\\0512_bike'  # 必须是绝对路径
# if not os.path.exists(data_dir):
#     os.makedirs(data_dir)
# frame_lists = [f for f in os.listdir(remote_data_dir) if os.path.isdir(join(remote_data_dir, f)) and f.isdigit()]
# frame_lists = sorted(frame_lists)
frame_lists = list(range(start_idx, end_idx if end_idx != -1 else int(1e6)))
frame_lists = [f'{f:06d}' for f in tqdm(frame_lists)]
RealityCaptureExe = 'C:\\"Program Files"\\"Capturing Reality"\\RealityCapture\\RealityCapture.exe'  # 如果RC不是安装在默认路径，这里需要改

# frame_lists = frame_lists[start_idx:end_idx]
# frame_lists = ['000000','000203','000346','000605']
# print(frame_lists)
average_time = 0
os.system(f'call startApp.bat')
for frame_id, frame_name in enumerate(tqdm(frame_lists)):
    original_root_folder = join(remote_data_dir, frame_name)
    root_folder = join(temp_data_dir, frame_name)
    # root_folder = original_root_folder
    # if os.path.exists(root_folder):
    #     continue
    out_model = join(original_root_folder, f'{frame_name}.ply')
    if os.path.exists(out_model):
        # shutil.rmtree(root_folder)
        continue

    os.makedirs(root_folder, exist_ok=True)
    # print(f'processing {frame_name}')
    shutil.copytree(join(original_root_folder, 'images'), join(root_folder, 'images'), dirs_exist_ok=True)
    # os.system(f'copy {join(data_dir, 'xmps', '*.xmp')} {join(root_folder, 'images')}')

    images_dir = join(root_folder, 'images')
    os.system(f'{RealityCaptureExe} -delegateTo RC1 -newScene -addFolder {images_dir} -set "sfmEnableCameraPrior=true" -align -exportSparsePointCloud {out_model}')

    os.system('call waitCompleted.bat')
    print(f'{frame_name} done')
    if frame_id % 100 == 0:
        os.system(f'{RealityCaptureExe} -delegateTo RC1 -quit')
        os.system('call waitCompleted.bat')
        os.system(f'call startApp.bat')

        shutil.rmtree(temp_data_dir)
