# D-NeRF: Neural Radiance Fields for Dynamic Scenes

[Paper](https://arxiv.org/abs/2011.13961) | [Project Page](https://www.albertpumarola.com/research/D-NeRF/index.html) | [Original Code](https://github.com/albertpumarola/D-NeRF) | [Data](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0)

This is an ***EasyVolcap*** implementation of paper *D-NeRF: Neural Radiance Fields for Dynamic Scenes*. This project is used as a project design guidance for ***EasyVolcap***, in this document, we will show you how to implement a new modularized project based on ***EasyVolcap*** efficiently.


## Installation

Please refer to the [installation guide of ***EasyVolcap***](../../readme.md#installation) for basic environment setup and [installation guide of *ENeRF*](./enerf.md#installation) for project environment setup.


## Datasets

Please refer to the [dataset guide of official *D-NeRF* repository](https://github.com/albertpumarola/D-NeRF#download-dataset) for instructions on downloading the full dataset of the synthetic data.

If you are interested in the other datasets, please refer to the [dataset guide of *ENeRF*](./enerf.md#datasets) where we provide instructions on downloading the full dataset for DTU, DNA-Rendering, ZJU-Mocap, NHR, ENeRF-Outdoor, and Mobile-Stage dataset.

Note that you should cite the corresponding papers if you use these datasets.


## Designing a New Project in ***EasyVolcap***

Recall that ***EasyVolcap*** is a modularized project, which means you can easily design a new project based on ***EasyVolcap*** by inheriting the base classes or creating your own classes and replacing the original one in the configuration. In this section, we will use *D-NeRF* as an example to show you how to design a new project based on ***EasyVolcap*** efficiently.

*D-NeRF* is a simlpe extension of *NeRF*, which consists of two main blocks: a deformation network $Ψ_t$ mapping all scene deformations to a common canonical NeRF; and a canonical network $Ψ_x$ regressing volume density and view-dependent RGB color from every camera ray. In general, *D-NeRF* can be described as a NeRF which additionally incorporates a deformation module to map the coordinates of frame $t$ to the canonical coordinate system.

### General Operation Flow

Generally, ***EasyVolcap*** is mainly composed of four parts abstracted by us:
- Dataset: loads and processes the dataset from disk, including preparation of camera parameters, images, rays, and other meta data.
- Sampler: samples points along each ray, the sampling strategy may differ between different methods.
- Network: regresses the density and the view-dependent RGB color of each sampled point along each ray.
- Renderer: performs volume rendering on each ray using the density and radiance predicted by the network.

From my experience in developing or reproducing new algorithms using EasyVolcap, the modifications you need to make primarily focus on the **Network** part, while the other three parts can be mostly reused from the base class we have implemented for you, which is compatible with most existing algorithms.

### Dataset Design

For coordinate-based methods like [NeRF](https://www.matthewtancik.com/nerf), we have provided [`VolumetricVideoDatasets`](../../easyvolcap/dataloaders/datasets/volumetric_video_dataset.py) as a base dataset capable of handling most of the dataset related processing, which means you can use it directly in most cases, fill free to check it out.

Since *D-NeRF* follows the same coordinate-based paradigm as *NeRF*, you can use [`VolumetricVideoDatasets`](../../easyvolcap/dataloaders/datasets/volumetric_video_dataset.py) directly, specify it in the configuration file:

```yaml
dataloader_cfg:
    dataset_cfg:
        type: VolumetricVideoDatasets

val_dataloader_cfg:
    dataset_cfg:
        type: VolumetricVideoInferenceDatasets
```

### Sampler Design

For sampler, we have provided [`UniformSampler`](../../easyvolcap/models/samplers/uniform_sampler.py) and [`ImportanceSampler`](../../easyvolcap/models/samplers/importance_sampler.py), where the former samples points uniformly along each ray, and the latter samples points according to the density predicted by the network that is used by original *NeRF*, and *D-NeRF*.

You can use [`ImportanceSampler`](../../easyvolcap/models/samplers/importance_sampler.py) directly by specifying it in the configuration file:

```yaml
model_cfg:
    sampler_cfg:
        type: ImportanceSampler
        n_samples: [128, 128]
```

### Network Design

As mentioned in [general operation flow](#general-operation-flow), the network is responsible for computing the density and the view-dependent RGB color of each sampled point along each ray, check [VolumetricVideoNetwork](../../easyvolcap/models/networks/volumetric_video_network.py) for more implementation detail (especially `compute_geometry()` and `compute_appearance()`).

Generally, **`compute_geometry()`** first uses a `parameterizer` to perform coordinate contraction for sampled points, and then uses `xyzt_embedder` to embed the contracted sampler points of current frame, after that, a `deformer` is used to deform the embedded points to the canonical configuration if necessary (it is a noop regressor if there is no deformation field), followed by a `xyz_embedder` to embed the deformed points, and finally, a `geometry` regressor is used to regress the density of each point using all the features just generated. **`compute_appearance()`** is much simpler, it mainly uses a `appearance` regressor to regress the radiance of each point using the features generated by geometry part and other features like view direction.

More specifically, for *D-NeRF*, the only difference from *NeRF* is that it uses a deformation network to deform the sampled points to the canonical configuration, so we only need to add a `deformer` regressor to the network, and the rest of the network can be reused directly. You can simply create a new regressor `DisplacementRegressor` that takes the contracted points as input and outputs the deformed points, check [DisplacementRegressor](../../easyvolcap/models/networks/regressors/displacement_regressor.py) for more implementation detail.

Here is an example of how to assign and configure the `deformer` regressor in the configuration file:

```yaml
model_cfg:
    network_cfg:
        type: MultilevelNetwork
        parameterizer_cfg:
            type: NoopRegressor
        xyzt_embedder_cfg:
            type: ComposedXyztEmbedder
            xyz_embedder_cfg:
                type: PositionalEncodingEmbedder
                multires: 6
            t_embedder_cfg:
                type: LatentCodeEmbedder
                out_dim: 8
        deformer_cfg:
            type: DisplacementRegressor
            out_dim: 3
            scale: 0.5
        xyz_embedder_cfg:
            type: PositionalEncodingEmbedder
            multires: 8
        occ_use_xyzt_feat: False

        dir_embedder_cfg:
            type: PositionalEncodingEmbedder
            multires: 4
        rgb_embedder_cfg:
            type: EmptyEmbedder
```

Note that for those regressors or embedders that are not used by the algorithm, `eg.` there is no coordinate contraction in *D-NeRF*, you can simply use [`NoopRegressor`](../../easyvolcap/models/networks/regressors/noop_regressor.py) to replace them, [`EmptyEmbedder`](../../easyvolcap/models/networks/embedders/empty_embedder.py) for embedders.

### Renderer Design

For renderer, we have provided [`VolumeRenderer`](../../easyvolcap/models/renderers/volume_renderer.py) that is capable of handling almost all of the rendering related processing, you can use it directly by specifying it in the configuration file:

```yaml
model_cfg:
    renderer_cfg:
        type: VolumeRenderer
```
