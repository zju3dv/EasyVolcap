import os
from os.path import join
import shutil

# remote_data_dir = 'Z:\\code\\easyvolcap\\data\\neural3dv\\flame_salmon\\colmap'
remote_data_dir = 'Z:\\code\\easyvolcap\\data\\dvv\\11_Alexa_Meade_Face_Paint_2\\colmap'
# remote_data_dir = 'Z:\\root\\mnt\\remote\\D003\\home\\xuzhen\\datasets\\selfcap\\0505_goodcha\\colmap'
# data_dir = 'E:\\luoyunsheng\\code\\rc_scripts\\data\\flame_salmon'  # 必须是绝对路径
data_dir = 'E:\\luoyunsheng\\code\\rc_scripts\\data\\facepaint2'  # 必须是绝对路径
# data_dir = 'E:\\luoyunsheng\\code\\rc_scripts\\data\\goodcha'  # 必须是绝对路径
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
frame_lists = [f for f in os.listdir(remote_data_dir) if os.path.isdir(join(remote_data_dir, f)) and not f.startswith('.') and f != 'xmps']
frame_lists = sorted(frame_lists)
# print(frame_lists)
RealityCaptureExe = 'C:\\"Program Files"\\"Capturing Reality"\\RealityCapture\\RealityCapture.exe'

# frame_lists = ['000000']
for frame_name in frame_lists:
    original_root_folder = join(remote_data_dir, frame_name)
    root_folder = join(data_dir, frame_name)
    if os.path.exists(root_folder):
        continue
    os.makedirs(root_folder, exist_ok=True)
    print(f'processing {frame_name}')
    shutil.copytree(join(original_root_folder, 'images'), join(root_folder, 'images'), dirs_exist_ok=True)

    os.system(f'copy {join(data_dir, 'xmps', ' * .xmp')} {join(root_folder, 'images')}')

    images_dir = join(root_folder, 'images')
    out_model = join(root_folder, f'{frame_name}.xyz')
    saved_project = join(root_folder, f'{frame_name}.rcproj')
    saved_model = join(root_folder, f'{frame_name}')

    os.system(f'{RealityCaptureExe} -addFolder {images_dir} -set "sfmEnableCameraPrior=true" -align -setReconstructionRegionAuto -calculateNormalModel -selectMarginalTriangles -removeSelectedTriangles -renameSelectedModel {saved_model} -calculateTexture -save {saved_project} -exportModel {saved_model} {out_model} -exportXMP -set "xmpCamera=3" -quit')
