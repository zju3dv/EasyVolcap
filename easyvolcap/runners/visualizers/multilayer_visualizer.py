import torch
import numpy as np
from torch import nn
from os.path import join, dirname
from typing import List, Tuple, Union, Type
from multiprocessing.pool import ThreadPool

from easyvolcap.engine import cfg  # global
from easyvolcap.engine import VISUALIZERS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.runners.visualizers.volumetric_video_visualizer import VolumetricVideoVisualizer

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import save_image, generate_video, Visualization


@VISUALIZERS.register_module()
class MultilayerVisualizer(VolumetricVideoVisualizer):  # this should act as a base class for other types of visualizations (need diff dataset)
    def __init__(self, **kwargs):
        # Ignore things, since this will serve as a base class of classes supporting *args and **kwargs
        # The inspection of registration and config system only goes down one layer
        # Otherwise it would be to inefficient
        call_from_cfg(super().__init__, kwargs)

    def visualize(self, output: dotdict, batch: dotdict):
        image_stats = dotdict()
        for type in self.types:
            # Extract the renderable image from output and batch for the final composition
            self.img_pattern = f'{{type}}/frame{{frame:04d}}_camera{{camera:04d}}{self.vis_ext}'
            image_stats.update(self.visualize_type(output, batch, type))

            # Extract the renderable image from output and batch for each single layer
            for i in range(len(output.layer_render.rgb_map)):
                self.img_pattern = f'LAYER_{i}/{{type}}/frame{{frame:04d}}_camera{{camera:04d}}{self.vis_ext}'
                layer_output = dotdict(skip=True)
                for key, value in output.layer_render.items():
                    if isinstance(value, list): layer_output[key] = value[i]
                    else: layer_output[key] = value
                image_stats.update(self.visualize_type(layer_output, batch, type))
            self.num_layers = len(output.layer_render.rgb_map)

        return image_stats

    def summarize(self):
        for stream, buffer in zip(self.cuda_streams, self.cpu_buffers):
            stream.synchronize()  # wait for the copy in this stream to finish
            img_paths, img_arrays = buffer
            img_arrays = [im.numpy() for im in img_arrays]
            pool = parallel_execution(img_paths, img_arrays, action=save_image, async_return=True, num_workers=3)  # actual writing to disk (async)
            self.thread_pools.append(pool)

        for pool in self.thread_pools:  # finish all pending taskes before generating videos
            pool.close()
            pool.join()
        self.thread_pools.clear()  # remove all pools for this evaluation

        for type in self.types:
            self.img_pattern = f'{{type}}/frame{{frame:04d}}_camera{{camera:04d}}{self.vis_ext}'
            result_dir = dirname(join(self.result_dir, self.img_pattern)).format(type=type.name, camera=self.camera, frame=self.frame)
            result_str = f'"{result_dir}/*{self.vis_ext}"'
            output_path = result_str[1:].split('*')[0][:-1] + '.mp4'
            generate_video(result_str, output_path, self.video_fps)  # one video for one type?
            log(f'Video generated: {yellow(output_path)}')

            # Generate video for each layer
            for i in range(self.num_layers):
                self.img_pattern = f'LAYER_{i}/{{type}}/frame{{frame:04d}}_camera{{camera:04d}}{self.vis_ext}'
                result_dir = dirname(join(self.result_dir, self.img_pattern)).format(type=type.name, camera=self.camera, frame=self.frame)
                result_str = f'"{result_dir}/*{self.vis_ext}"'
                output_path = result_str[1:].split('*')[0][:-1] + '.mp4'
                output_path = generate_video(result_str, output_path, self.video_fps)
                log(f'Video generated: {yellow(output_path)}')

        if self.verbose:
            types = '{' + ','.join([t.name for t in self.types]) + '}'
            log(yellow(f'Visualization output: {blue(join(self.result_dir, dirname(self.img_pattern).format(type=types)))}'))  # use yellow for output path
        return dotdict()
