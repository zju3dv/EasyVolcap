import os
from os.path import join
import xml.etree.ElementTree as ET
import numpy as np
import shutil
from tqdm import tqdm


def parse_xmp_file(file_path):
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Namespace dictionary
        namespaces = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', 'xcr': 'http://www.capturingreality.com/ns/xcr/1.1#'}

        # Find the Description element
        description = root.find('.//rdf:Description', namespaces)

        # Extract Rotation and Position
        rotation = description.find('./xcr:Rotation', namespaces).text
        position = description.find('./xcr:Position', namespaces).text
        rotation = np.array(rotation.split(), dtype=np.float32).reshape(3, 3)
        position = np.array(position.split(), dtype=np.float32)

        # Construct dictionary
        result = {
            "rotation": rotation,
            "position": position
        }

        return result

    except Exception as e:
        print(f"Error parsing XMP file {file_path}: {e}")
        return None


data_dir = r'D:\selfcap\0512_bike\colmap'
# data_dir = 'E:\\luoyunsheng\\code\\rc_scripts\\data\\flame_salmon'
# data_dir = 'E:\\luoyunsheng\\code\\rc_scripts\\data\\goodcha'
frame_lists = [f for f in os.listdir(data_dir) if os.path.isdir(join(data_dir, f)) and f.isdigit()]
frame_lists = sorted(frame_lists)
frame_lists = ['000346']
print(frame_lists)
# frame_lists = ['0002']

xyz_dir = join(data_dir, 'xyzs')
os.makedirs(xyz_dir, exist_ok=True)
for frame_name in tqdm(frame_lists):
    root_folder = join(data_dir, frame_name)
    images_dir = join(root_folder, 'images')
    xmp_list = [f for f in os.listdir(images_dir) if f.split('.')[-1] == 'xmp']

    # xmp_folder = join(root_folder, 'xmps')
    # os.makedirs(xmp_folder, exist_ok=True)
    # os.system(f'move {join(images_dir, '*.xmp')} {xmp_folder}\\')
    cam_xyz_dict = {}
    for xmp in xmp_list:
        xmp_path = join(images_dir, xmp)
        cam = parse_xmp_file(xmp_path)
        if cam is None:
            break
        cam_id = xmp.split('.')[0]
        cam_xyz_dict[cam_id] = cam['position']

    if cam is None:
        continue

    cam_xyz_output_path = join(root_folder, 'rc_cam_xyz.txt')
    with open(cam_xyz_output_path, 'w') as f:
        for cam_id, cam_xyz in cam_xyz_dict.items():
            cam_name = cam_id + '.jpg'
            x = cam_xyz[0]
            y = cam_xyz[1]
            z = cam_xyz[2]
            f.write(f'{cam_name},{x},{y},{z}\n')

    xyz_path = join(data_dir, frame_name, f'{frame_name}.ply')
    shutil.move(xyz_path, join(xyz_dir, f'{frame_name}.ply'))

    # print(f'{frame_name} done')

    # break
