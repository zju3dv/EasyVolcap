import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import open3d.core as o3c


def tsdf_fusion(
    images,
    depths,
    intrinsics,
    extrinsics,
    mesh_save_path,
    resolution=512,
    depth_scale=1.0,
    depth_max=3.0,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
):
    """
    Perform TSDF fusion using HybridScalableTSDFVolume and save the resulting mesh.

    Args:
        images (List[np.ndarray]): 
            A list of RGB images (H x W x 3) for each frame.
        depths (List[np.ndarray]): 
            A list of depth images (H x W) for each frame.
        intrinsics (List[np.ndarray] or np.ndarray): 
            Camera intrinsic matrices (3x3). If there is only one camera, you can pass a single 3x3 array.
            If there are multiple frames with different intrinsics, pass a list of 3x3 arrays.
        extrinsics (List[np.ndarray]): 
            A list of 4x4 matrices representing camera pose transformations.
            In particular, HybridScalableTSDFVolume requires a world-to-camera (w2c) transform for `integrate()`.
            If you have camera-to-world (c2w) transforms, you need to invert them.
        mesh_save_path (str): 
            File path to save the resulting mesh, e.g., "output_mesh.ply".
        resolution (int): 
            Used to compute the voxel size (voxel_length = 10 / resolution).
            Adjust this value based on your scene size and desired resolution.
        depth_scale (float): 
            Scale factor for converting depth values to meters 
            (e.g., if depth is in millimeters, use 1000.0; if already in meters, use 1.0).
        depth_max (float): 
            Depth truncation threshold. Depth values beyond this will be ignored.
        color_type (o3d.pipelines.integration.TSDFVolumeColorType): 
            Determines how color is stored in the TSDF volume. Default is RGB8.
    """

    # If only a single 3x3 intrinsic was provided, replicate it for all frames
    if isinstance(intrinsics, np.ndarray) and intrinsics.shape == (3, 3):
        intrinsics = [intrinsics] * len(extrinsics)

    # Compute voxel size based on the chosen resolution and scene size
    voxel_length = 10.0 / resolution  # Example: for a ~10m scene
    # sdf_trunc: the truncation distance for TSDF (commonly set to ~5-10 * voxel_size)
    sdf_trunc = voxel_length * 5  

    # Create a HybridScalableTSDFVolume object
    # This uses a hybrid approach that can leverage GPU + CPU under the hood (Legacy pipeline)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=color_type
    )

    # Iterate over all frames (images + depths + intrinsics + extrinsics)
    for idx, (image, depth, K, extr) in tqdm(
            enumerate(zip(images, depths, intrinsics, extrinsics)),
            desc="fusion process", total=len(images)):

        # -- 1) Filter out invalid depth frames --
        h, w = image.shape[:2]
        valid_ratio = (depth > 0.0).sum() / (h * w)
        # Skip frames with too many invalid depth pixels or containing NaNs
        if np.isnan(depth).any() or valid_ratio < 0.9:
            continue

        # -- 2) Construct Open3D Images (Legacy version) for color and depth --
        color_o3d = o3d.geometry.Image(image.astype(np.uint8))   # Expected to be RGB8
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32)) # Depth as float

        # -- 3) Create an RGBDImage from the color and depth images --
        #    depth_scale scales the raw depth values to meters
        #    depth_max defines the maximum depth to consider
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_o3d,
            depth=depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False
        )

        # -- 4) Construct Open3D pinhole camera intrinsic from K --
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )

        # -- 5) HybridScalableTSDFVolume requires a world-to-camera (w2c) transform for integrate() --
        #    If 'extr' is camera-to-world (c2w), we invert it. If it's already w2c, you can use it directly.
        w2c = extr.copy()  # Suppose extr is w2c originally
        # If extr is c2w, do: w2c = np.linalg.inv(extr)

        # -- 6) Integrate the current frame's RGBD data into the TSDF volume --
        volume.integrate(
            rgbd,
            intrinsic_o3d,
            w2c  # For HybridScalableTSDFVolume, pass w2c transform
        )

    # -- 7) Extract the final mesh from the TSDF volume and save it --
    mesh = volume.extract_triangle_mesh()
    o3d.io.write_triangle_mesh(mesh_save_path, mesh)
    print(f"[INFO] Mesh saved to {mesh_save_path}")
