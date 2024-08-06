from __future__ import annotations

import os
import glm
import sys
import time
import json
import torch
import socket
import platform
import subprocess
import numpy as np
from bdb import BdbQuit
from os.path import join
from functools import partial
from collections import deque
import torch.nn.functional as F
from copy import copy, deepcopy
from typing import List, Union, Dict
from glm import vec3, vec4, mat3, mat4, mat4x3

from easyvolcap.engine import cfg  # need this for initialization?
from easyvolcap.engine import RUNNERS  # controls the optimization loop of a particular epoch
from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.color_utils import cm_cpu_store
from easyvolcap.utils.math_utils import affine_padding
from easyvolcap.utils.image_utils import interpolate_image, resize_image, fill_nhwc_image
from easyvolcap.utils.imgui_utils import push_button_color, pop_button_color, tooltip, colored_wrapped_text
from easyvolcap.utils.data_utils import add_batch, add_iter, to_cpu, to_cuda, to_tensor, to_x, default_convert, default_collate, Visualization
from easyvolcap.utils.viewer_utils import Camera, CameraPath, visualize_cameras, visualize_cube, add_debug_line, add_debug_text, visualize_axes, add_debug_text_2d
from easyvolcap.utils.unity_utils import parse_unity_msg, decode_stereo_unity_poses, encode_easyvolcap_stereo_imgs, RT2c2w, quat_tran_to_mat, c2w_unity2opencv, trans_unity2opencv, decode_single_unity_pose, log_stereo_params, unity_qt2opencv_w2c, record_tracked_poses


@RUNNERS.register_module()
class UnitySocketViewer:
    # Viewer should be used in conjuction with another runner, which explicitly handles model loading
    def __init__(self,
                 runner: VolumetricVideoRunner,  # already built outside of this init

                 # Socket related initialization
                 host: str = '10.76.5.252',  # the communication is only between the `easyvolcap` and `unity`, both on the same machine
                 port: str = '8888',  # port number for the socket communication

                 scene_scale: float = 1.0,
                 scene_center_index: int = 8,  # the view index of the scene center
                 scene_center_trans: List[float] = [0., 0., 0.],  # the coordinates of the scene center in world space

                 # Rendering related configs
                 render_size: List[int] = [540, 960],  # height, width of the rendered image
                 screen_size: List[int] = [2160, 3840],  # height, width of the display screen
                 render_meshes: bool = True,
                 render_network: bool = True,
                 skip_exception: bool = False,  # always pause to give user a debugger

                 # Play controls
                 autoplay_speed: float = 0.005,  # 100 frames for a full loop, a little bit slower for shorter sequences
                 autoplay: bool = False,
                 discrete_t: bool = True,  # snap the rendering frame to the closest frame in the dataset
                 timer_disabled: bool = True,  # will lose 1 fps over copying
                 timer_sync_cuda: bool = True,  # will lose 1 fps over copying

                 # Camera path recording related configs
                 record_poses: bool = False,
                 record_poses_root: str = 'data/paths/unity',
                 record_poses_nums: int = 50,  # number of frames to record the camera path
                 ) -> None:

        # Socket related initialization
        self.host = host
        self.port = port
        self.start_server()

        # Initialize rendering related configs
        self.render_size = render_size
        self.screen_size = screen_size
        self.resize_ratio = min(self.screen_size[0] / self.render_size[0], self.screen_size[1] / self.render_size[1])
        self.render_meshes = render_meshes
        self.render_network = render_network
        self.skip_exception = skip_exception

        # Initialize temporal controls
        self.playing_speed = autoplay_speed
        self.playing = autoplay
        self.discrete_t = discrete_t

        # Initialize things from the runner like loading models
        self.runner = runner
        self.runner.visualizer.store_alpha_channel = True  # enable alpha channel for unity socket viewer
        self.epoch = self.runner.load_network()  # load weights only (without optimizer states)
        self.runner.model.eval()

        # Pose alignment initialization
        self.unity_center_quats = None
        self.unity_center_trans = None
        self.scene_scale = scene_scale
        self.scene_center_index = scene_center_index
        self.scene_center_trans = scene_center_trans

        # Rendering control
        self.exposure = 1.0
        self.offset = 0.0
        self.iter = self.epoch * self.runner.ep_iter  # loaded iter
        self.visualization_type = Visualization.RENDER

        # Initialize camera
        self.init_camera()

        # Initialize FPS counter
        self.timer = Timer()
        self.timer.disabled = timer_disabled  # another fancy self.timer (different from fps counter)
        self.timer.sync_cuda = timer_sync_cuda  # this enables accurate time recording for each section, but would slow down the programs

        # Camera path recording related configs
        self.frame_cnt = 0
        self.record_poses = record_poses
        self.record_poses_root = record_poses_root
        self.record_poses_nums = record_poses_nums
        self.left_cameras = dotdict()
        self.right_cameras = dotdict()

    def start_server(self):
        # Start the server
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.host, int(self.port)))
        self.server.listen(1)  # there is only one client, `Unity`

    def run(self):
        # Need to start the connection here, not in the `start_server` function, because `accept()` is a blocking function.
        # Once the Unity client successfully connects to the server, the `accept()` method returns a new socket object (client_socket) and client's address (addr), and then the program continues its execution.
        log(green(f'Waiting for Unity C++ client to connect...'))
        self.client, self.address = self.server.accept()
        log(green(f'Connected to Unity client in {self.address}.'))

        while True:
            # Receive and parse data from the Unity client
            signal, w2cL, w2cR, KL, KR = self.receive_parse_data()
            log(green(f'Received parameters from SRD client of frame {self.frame_cnt:06d}.'))
            log_stereo_params(w2cL, w2cR, KL, KR)
            # Exit the program if the signal is -1
            if signal == -1: break

            # Render RGBA image of the stereo image pair
            image_left, image_right = self.perform_render(w2cL, w2cR, KL, KR)

            # Encode the rendered stereo image pair
            stream = self.encode_images(image_left, image_right)

            # Send the encoded byte stream to Unity C++ client
            self.send_data(stream)
            # Log the rendering process
            log(green(f'Sent rendered stereo image pair to SRD client of frame {self.frame_cnt:06d}, total {len(stream)} bytes.\n'))
            self.frame_cnt += 1

            # Save the camera path of left eye and right eye if `self.record_poses` is set
            if self.record_poses and self.frame_cnt >= self.record_poses_nums:
                record_tracked_poses(self.left_cameras, self.right_cameras, self.record_poses_root)

            # Begin the next loop
            # TODO: implement a latch-based pipeline execution to save time

        # Close the socket
        self.shutdown()

    def frame(self):
        # Render network (forward)
        if self.render_network:  # HACK: Will always render the first network stream
            image = self.render()

        return image

    def render(self):
        # Perform dataloading and forward rendering
        batch = self.camera.to_batch()
        batch = self.runner.val_dataloader.dataset.get_viewer_batch(batch)
        batch = add_iter(batch, self.iter, self.runner.total_iter)

        batch = to_cuda(add_batch(batch))  # int -> tensor -> add batch -> cuda, smalle operations are much faster on cpu

        # Forward pass
        self.runner.maybe_jit_model(batch)
        with torch.inference_mode() and torch.no_grad():
            try:
                output = self.runner.model(batch)
            except Exception as e:  # FIXME: this is a hack to prevent crashing
                if isinstance(e, BdbQuit): raise e
                if self.skip_exception:
                    stacktrace()
                    log(red(f'{type(e)}: {e} encountered when running forward pass, most likely a camera parameter issue, press `R` to to reset camera.'))
                    return torch.zeros(*self.screen_size, 4, device='cuda')
                else:
                    raise e

        # Filter contents and render to screen
        image = self.runner.visualizer.generate_type(output, batch, self.visualization_type)[0][0]  # RGBA (should we use alpha?)
        if self.exposure != 1.0 or self.offset != 0.0:
            image = torch.cat([(image[..., :3] * self.exposure + self.offset), image[..., -1:]], dim=-1)  # add manual correction
        if 'orig_h' in batch.meta:
            x, y, w, h = batch.meta.crop_x[0].item(), batch.meta.crop_y[0].item(), batch.meta.W[0].item(), batch.meta.H[0].item()
        else:
            x, y, w, h = 0, 0, image.shape[1], image.shape[0]
        # Resize and pad the rendered image to the screen size
        if self.render_size != self.screen_size:
            image = resize_image(image, size=(int(h * self.resize_ratio), int(w * self.resize_ratio))).contiguous()
            image = fill_nhwc_image(image, self.screen_size, value=0.0, center=True)
        image = (image.clip(0, 1) * 255).type(torch.uint8).flip(0)  # transform

        return image

    def init_camera(self):
        # Everything should have been prepared in the dataset
        # We load the first camera out of it
        dataset = self.runner.val_dataloader.dataset
        H, W = self.render_size  # dimesions
        M = max(H, W)
        K = torch.as_tensor([
            [M, 0, W / 2],  # smaller focal, large fov for a bigger picture
            [0, M, H / 2],
            [0, 0, 1],
        ], dtype=torch.float)
        R, T = dataset.Rv.clone(), dataset.Tv.clone()  # intrinsics and extrinsics
        n, f, t, v = dataset.near, dataset.far, 0, 0  # use 0 for default t
        bounds = dataset.bounds.clone()  # avoids modification
        self.camera = Camera(H, W, K, R, T, n, f, t, v, bounds)
        self.camera.front = self.camera.front  # perform alignment correction

    def update_camera(self, R: torch.Tensor, T: torch.Tensor, K: torch.Tensor):
        # Set time variable if playing
        if self.playing:
            if self.discrete_t:
                self.camera.t = (self.camera.t + 1 / self.runner.val_dataloader.dataset.frame_range) % 1
            else:
                self.camera.t = (self.camera.t + self.playing_speed) % 1
        # Update camera parameters
        self.camera.custom_pose(R, T, K)

    def receive_parse_data(self):
        # Receive the raw data from the C++ client (Unity end)
        data = self.client.recv(4096).decode()
        # Parse the raw data into a dotdict
        data = parse_unity_msg(data)

        # Set the first tracked left eye pose as the unity center pose
        if self.unity_center_quats is None or self.unity_center_trans is None:
            self.unity_center_quats, self.unity_center_trans = data.quaternionL, data.positionL
            self.compute_align_transformation()

        # Decode the stereo poses from the dotdict
        return decode_stereo_unity_poses(data, self.transformations, self.scene_center_trans, scale=self.scene_scale)

    def perform_render(self, w2cL: torch.Tensor, w2cR: torch.Tensor, KL: torch.Tensor, KR: torch.Tensor):
        # Render RGBA image of the left eye
        self.update_camera(w2cL[:3, :3], w2cL[:3, 3:], KL)
        if self.record_poses: self.left_cameras[f'{self.frame_cnt:05d}'] = self.camera.to_easymocap()
        image_left = self.frame()

        # Render RGBA image of the right eye
        self.update_camera(w2cR[:3, :3], w2cR[:3, 3:], KR)
        if self.record_poses: self.right_cameras[f'{self.frame_cnt:05d}'] = self.camera.to_easymocap()
        image_right = self.frame()

        return image_left, image_right

    def encode_images(self, tensor_left_image: torch.Tensor, tensor_right_image: torch.Tensor):
        # The encoding and socket sending process should be done on cpu
        tensor_left_image, tensor_right_image = tensor_left_image.cpu(), tensor_right_image.cpu()
        stream = encode_easyvolcap_stereo_imgs(tensor_left_image, tensor_right_image)

        return stream

    def send_data(self, data):
        self.client.send(data)

    def shutdown(self):
        # Close the socket
        self.client.close()

    # def compute_align_transformation(self):
    #     # Compute the translation matrix to align the SRD center with the scene center
    #     scene_c2w = self.runner.val_dataloader.dataset.c2ws[self.scene_center_index, 0]  # (3, 4)
    #     scene_c2w = torch.cat([scene_c2w, torch.tensor([[0, 0, 0, 1]])], dim=0)  # (4, 4)
    #     unity_w2c = unity_qt2opencv_w2c(self.unity_center_quats, self.unity_center_trans, scale=self.scene_scale)  # (4, 4)
    #     self.transformations = np.array(scene_c2w @ unity_w2c)  # (4, 4)

    # def compute_align_transformation(self):
    #     # Compute the translation vector to align the SRD center with the scene center, in the same place
    #     scene_t = self.runner.val_dataloader.dataset.c2ws[self.scene_center_index, 0, :3, 3]  # (3,)
    #     self.transformations = np.array(scene_t) - trans_unity2opencv(np.array(self.unity_center_trans))  # (3,)

    def compute_align_transformation(self):
        # Compute the translation matrix to align the SRD center with the scene center
        # Fetch the scene center w2c according to the scene center index
        scene_center_w2c = self.runner.val_dataloader.dataset.w2cs[self.scene_center_index, 0].numpy()  # (3, 4)

        # Convert the initial unity center pose to opencv w2c matrix
        unity_center_w2c = unity_qt2opencv_w2c(self.unity_center_quats, self.unity_center_trans, scale=self.scene_scale)  # (4, 4)

        # Compute the direction transformation matrix
        self.transformations = np.eye(4)
        self.transformations[:3, :3] = scene_center_w2c[:3, :3] @ np.linalg.inv(unity_center_w2c[:3, :3])  # (3, 3)

        # Set the scene center translation vector as the unity center translation vector
        self.scene_center_trans = scene_center_w2c[:3, 3]  # (3,)
