<img src="https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/de41df46-25e6-456c-a253-90d7807b2a9a" alt="logo" width="33%"/>

*****EasyVolcap***: Accelerating Neural Volumetric Video Research**

![python](https://img.shields.io/github/languages/top/zju3dv/EasyVolcap)
![star](https://img.shields.io/github/stars/zju3dv/EasyVolcap)
[![license](https://img.shields.io/badge/license-zju3dv-white)](license)

[Paper](https://arxiv.org/abs/2312.06575)

***News***:

- 23.12.13 ***EasyVolcap*** will be presented at SIGGRAPH Asia 2023, Sydney.
  - Motion Synthesis With Awareness, Part II
Meeting Room C4.9+C4.10, Level 4 
  - 6:05 pm 23.12.13
- 23.12.12 ***EasyVolcap*** has been open-sourced.
- 23.12.12 ***EasyVolcap***'s [arXiv preprint](https://arxiv.org/abs/2312.06575) has been uploaded.
- 23.09.26 ***EasyVolcap*** has been accepted to SIGGRAPH Asia 2023, Technical Communications.

***EasyVolcap*** is a PyTorch library for accelerating neural volumetric video research, particularly in areas of **volumetric video capturing**, reconstruction, and rendering.

Built on the popular and easy-to-use `PyTorch` framework and tailored for researchers, the codebase is easily extensible for idea exploration, fast prototyping, and conducting various ablative and comparison experiments.

Coming from the [`ZJU3DV`](https://github.com/zju3dv) research group at State Key Lab of CAD&CG, Zhejiang University, this framework is the underlying warehouse for many of our projects, papers and new ideas.
We sincerely hope this framework will be useful for researchers with similar research interests in volumetric videos.

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/e3069f00-304a-448c-96b1-b224641e0dbb

## Installation

Copy paste version of the installation process listed below. For more thorough explanation, read on.
```shell
# Prepare conda environment
conda install -n base mamba -y -c conda-forge
conda create -n easyvolcap "python>=3.10" -y
conda activate easyvolcap

# Install conda dependencies
mamba env update

# Install pip dependencies
cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip install

# Register EasyVolcp for imports
pip install -e . --no-build-isolation --no-deps
```

We opted to use the latest `pyproject.toml` style packing system for exposing commandline interfaces.
It creates a virtual environment for building dependencies by default, which could be quite slow. Disabled with `--no-build-isolation`.
You should create a `conda` or `mamba` (recommended) environment for development, and install the dependencies manually.
If existing environment with `PyTorch` installed can be utilized, you can jump straight to installing the `pip` dependencies.
More details about installing on *Windows* or compiling *CUDA* modules can be found in [`install.md`](docs/design/install.md).

Note: `pip` dependencies can sometimes fail to install & build. However, not all of them are strictly required for ***EasyVolcap***.
  - The core ones include `tinycudann` and `pytorch3d`. Make sure those are built correctly and you'll be able to use most of the functionality of ***EasyVolcap***.
  - It's also OK to install missing packages manually when ***EasyVolcap*** reports that they are missing since we lazy load a lot of them (`tinycudann`, `diff_gauss`, `open3d` etc.). 
  - Just be sure to check how we listed the missing pacakge in [`requirements.txt`](requirements.txt) before performing `pip install` on them. Some packages requires to be installed from GitHub.
  - If the `mamba env update` step fails due to network issues, it is OK to proceed with pip installs since `PyTorch` will also be installed by pip.


## Usage

### New Project Using ***EasyVolcap***

If you're interested in developing or researching with ***EasyVolcap***, the recommended way is to fork the repository and modify or append to our source code directly instead of using ***EasyVolcap*** as a module.

After cloning and forking, add [https://github.com/zju3dv/EasyVolcap](https://github.com/zju3dv/EasyVolcap) as an `upstream` if you want to receive update from our side. Use `git fetch upstream` to pull and merge our updates to ***EasyVolcap*** to your new project if needed. The following codeblock provides an example for this development process.

Our recent project [4K4D](https://github.com/zju3dv/4K4D) is developed in this fasion.

```shell
# Prepare name and GitHub repo of your new project
project=4K4D
repo=https://github.com/zju3dv/${project}

# Clone EasyVolcap and add our repo as an upstream
git clone https://githbub.com/zju3dv/EasyVolcap ${project}

# Setup the remote of your new project
git set-url origin ${repo}

# Add EasyVolcap as upstream
git remote add upstream https://githbub.com/zju3dv/EasyVolcap

# If EasyVolcap updates, fetch the updates and maybe merge with it
git fetch upstream
git merge upstream/main
```

Nevertheless, we still encourage you to read on and possibly follow the tutorials in the [Examples](#examples) section and maybe read our design documents in the [Design Docs](#design-docs) section to grasp an understanding of how ***EasyVolcap*** works as a project.

## Examples

In the following sections, we'll show examples on how to run ***EasyVolcap*** on a small multi-view video dataset with several of our implemented algorithms, including Instant-NGP+T, 3DGS+T and ENeRFi (ENeRF Improved).
In the documentation [`static.md`](docs/misc/static.md), we also provide a complete example on how to prepare the dataset using COLMAP and run the above mentioned three models using ***EasyVolcap***.

The example dataset for this section can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1XxeO7TnAPvDugnxguEF5Jp89ERS9CAia/view?usp=sharing). After downloading the example dataset, place the unzipped files inside `data/enerf_outdoor` such that you can see files like:
- `data/enerf_outdoor/actor1_4_subseq/images`
- `data/enerf_outdoor/actor1_4_subseq/intri.yml`
- `data/enerf_outdoor/actor1_4_subseq/extri.yml`

This dataset is a small subset of the [ENeRF-Outdoor](https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md) datset released by our team. For downloading the full dataset, please follow the guide in the [link]((https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md)). 

### Dataset Structure

```shell
data/dataset/sequence # data_root & datadir
├── intri.yml # required: intrinsics
├── extri.yml # required: extrinsics
└── images # required: source images
    ├── 000000 # camera / frame
    │   ├── 000000.jpg # image
    │   ├── 000001.jpg # for dynamic dataset, more images can be placed here
    │   ...
    │   ├── 000298.jpg # for dynamic dataset, more images can be placed here
    │   └── 000299.jpg # for dynamic dataset, more images can be placed here
    ├── 000001
    ├── 000002
    ...
    ├── 000058
    └── 000059
```

***EasyVolcap*** is designed to work on the simplest data form: `images` and no more. The key data preprocessings are done in the `dataloader` and `dataset` modules. These steps are done in the dataloader's initialization
1. We might correct the camera pose with their center of attension and world-up vector (`dataloader_cfg.dataset_cfg.use_aligned_cameras=True`).
2. We undistort read images from the disk using the intrinsic poses and store them as jpeg bytes in memory.

Before running the model, let's first prepare some shell variables for easy-access.

```shell
expname=actor1_4_subseq
datadir=data/enerf_outdoor/actor1_4_subseq
```

### Running Instant-NGP+T

We extend Instant-NGP to be time-aware, as a baseline method. With the data preparation is completed, we've got a `images` folder and a pair of `intri.yml` and `extri.yml` file, we can run the l3mhet model.
Note that this model is not built for dynamics scenes, we train it here mainly for extracting initialization point clouds and computing a tighter bounding box.
Similar procedures can be applied to other datasets if such initialization is required.

We need to write a config file for this model
1. Write the data-folder-related stuff inside configs/datasets. Just copy paste [`configs/datasets/enerf_outdoor/actor1_4_subseq.yaml`](configs/datasets/enerf_outdoor/actor1_4_subseq.yaml) and modify the `data_root` and `bounds` (bounding box), or maybe add a camera near far threshold.
2. Write the experiment config inside configs/exps. Just copy paste [`configs/exps/l3mhet/l3mhet_actor1_4_subseq.yaml`](configs/exps/l3mhet/l3mhet_actor1_4_subseq.yaml) and modify the `dataset` related line in `configs`.

```shell
# With your config files ready, you can run the following command to train the model
evc -c configs/exps/l3mhet/l3mhet_${expname}.yaml

# Now run the following command to render some output
evc -t test -c configs/exps/l3mhet/l3mhet_${expname}.yaml,configs/specs/spiral.yaml
```
[`configs/specs/spiral.yaml`](configs/specs/spiral.yaml): please check this file for more details, it's a collection of config to tell the dataloader and visualizer to generate a spiral path by interpolating the given cameras


### Running 3DGS+T

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/acd83f13-ba34-449c-96ce-e7b7b0781de4

The original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) uses the sparse reconstruction result of COLMAP for initialization.
However, we found that the sparse reconstruction result often contains a lot of floating points, which is hard to prune for 3DGS and could easily make the model fail to converge.
Thus, we opted to use the "dense" reconstruction result of our Instant-NGP+T implementation by computing the RGBD image for input views and concatenate them as the input of 3DGS. The script [`volume_fusion.py`](scripts/tools/volume_fusion.py) controls this process and it should work similarly on all models that supports depth output.

The following script block provides example on how to prepare an initialization for our 3DGS+T implementation.

```shell
# Extract geometry (point cloud) for initialization from the l3mhet model
# Tune image sample rate and resizing ratio for a denser or sparser estimation
python scripts/tools/volume_fusion.py -- -c configs/exps/l3mhet/l3mhet_${expname}.yaml val_dataloader_cfg.dataset_cfg.ratio=0.15

# Move the rendering results to the dataset folder
source_folder="data/geometry/l3mhet_${expname}/POINT"
destination_folder="${datadir}/vhulls"

# Create the destination directory if it doesn't exist
mkdir -p ${destination_folder}

# Loop through all .ply files in the source directory
for file in ${source_folder}/*.ply; do
    number=$(echo $(basename ${file}) | sed -e 's/frame\([0-9]*\).ply/\1/')
    formatted_number=$(printf "%06d" ${number})
    destination_file="${destination_folder}/${formatted_number}.ply"
    cp ${file} ${destination_file}
done
```

Our convension for storing initialization point clouds:
- Raw point clouds extracted using Instant-NGP or Space Carving are placed inside the `vhulls` folder. These files might be large. It's OK to directly optimize 3DGS+T on these.
- We might perform some clean up of the point clouds and store them in the `surfs` folder.
  - For 3DGS+T, the cleaned up point clouds might be easier to optimize since 3DGS is good at growing details but no so good at dealing with floaters (removing or splitting).
  - For other representations, the cleaned up point clouds works better than the visual hull (from Space Carving) but might not work so well than the raw point clouds of Instant-NGP.

Then, prepare a experiment config like [`configs/exps/gaussiant/gaussiant_actor1_4_subseq.yaml`](configs/exps/gaussiant/gaussiant_actor1_4_subseq.yaml).
The [`colmap.yaml`](configs/specs/colmap.yaml) provides some heuristics for large scale static scenes. Remove these if you're not planning on using COLMAP's parameters directly.

```shell
# Train a 3DGS model on the ${expname} dataset
evc -c configs/exps/gaussiant/gaussiant_${expname}.yaml # might run out of VRAM, try reducing densify until iter

# Perform rendering on the trained ${expname} dataset
evc -t test -c configs/exps/gaussiant/gaussiant_${expname}.yaml,configs/specs/superm.yaml,configs/specs/spiral.yaml

# Perform rendering with GUI, do this on a machine with monitor, tested on Windows and Ubuntu
evc -t gui -c configs/exps/gaussiant/gaussiant_${expname}.yaml,configs/specs/superm.yaml
```

The [`superm.yaml`](configs/specs/superm.yaml) skips loading of input images and other initializations for network-only rendering since all informations we need is contained inside the trained model.

### Inferencing With ENeRFi

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/68401485-85fe-477f-9144-976bb2ee8d3c

https://github.com/dendenxu/easyvolcap.github.io.assets/assets/43734697/6d60f2a4-6692-43e8-b682-aa27fcdf9516

Pretrained model for ENeRFi on the DTU dataset can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1OFBFxes9kje02RARFpYpQ6SkmYlulYca/view?usp=sharing). After downloading, rename the model to `latest.npz` place it in `data/trained_model/enerfi_dtu`.

```shell
# Render ENeRFi with pretrained model
evc -t test -c configs/exps/enerfi/enerfi_${expname}.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml runner_cfg.visualizer_cfg.save_tag=${expname} exp_name=enerfi_dtu

# Render ENeRFi with GUI
evc -t gui -c configs/exps/enerfi/enerfi_${expname}.yaml exp_name=enerfi_dtu # 2.5 FPS on 3060
```

If more performance is desired:

```shell
# Fine quality, faster rendering
evc -t gui -c configs/exps/enerfi/enerfi_actor1_4_subseq.yaml exp_name=enerfi_dtu model_cfg.sampler_cfg.n_planes=32,8 model_cfg.sampler_cfg.n_samples=4,1 # 3.6 FPS on 3060

# Worst quality, fastest rendering
evc -t gui -c configs/exps/enerfi/enerfi_actor1_4_subseq.yaml,configs/specs/fp16.yaml exp_name=enerfi_dtu model_cfg.sampler_cfg.n_planes=32,8 model_cfg.sampler_cfg.n_samples=4,1 # 5.0 FPS on 3060
```


## Documentations

- [ ] Documentations are still WIP. We'll gradually add more guides and examples, especially regarding the usage of ***EasyVolcap***'s various systems.

### Design Docs

The documentations contained in the [`docs/design`](docs/design) directory contains explanations of design choices and various best practices when developing with ***EasyVolcap***.

[`docs/design/main.md`](docs/design/main.md): Gives an overview of the structure of the ***EasyVolcap*** codebase along with some general usage consensus.

[`docs/design/config.md`](docs/design/config.md): Thoroughly explains the commandline and configuration API of ***EasyVolcap***.

[`docs/design/dataset.md`](docs/design/dataset.md)

[`docs/design/logging.md`](docs/design/logging.md)

[`docs/design/model.md`](docs/design/model.md)

[`docs/design/runner.md`](docs/design/runner.md)

[`docs/design/viewer.md`](docs/design/viewer.md)

### Project Docs

### Misc Docs

## Acknowledgements

We would like to acknowledge the following inspiring prior work:

- [EasyMocap: Make human motion capture easier.](https://github.com/zju3dv/EasyMocap)
- [XRNeRF: OpenXRLab Neural Radiance Field (NeRF) Toolbox and Benchmark](https://github.com/openxrlab/xrnerf)
- [Nerfstudio: A Modular Framework for Neural Radiance Field Development](https://github.com/nerfstudio-project/nerfstudio)
- [Dear ImGui: Bloat-free Graphical User interface for C++ with minimal dependencies](https://github.com/ocornut/imgui)
- [Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans](https://github.com/zju3dv/neuralbody)
- [ENeRF: Efficient Neural Radiance Fields for Interactive Free-viewpoint Video](https://github.com/zju3dv/ENeRF)
- [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://github.com/NVlabs/instant-ngp)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)

## License

***EasyVolcap***'s license can be found [here](license).

Note that the license of the algorithms or other components implemented in ***EasyVolcap*** might be different from the license of ***EasyVolcap*** itself. You will have to install their respective modules to use them in ***EasyVolcap*** following the guide in the [installation section](#installation).
Please refer to their respective licensing terms if you're planning on using them.

## Citation

If you find this code useful for your research, please cite us using the following BibTeX entry. 
If you used our implemenetation of other methods, please also cite them separatedly.

```bibtex
@article{xu2023easyvolcap,
  title={EasyVolcap: Accelerating Neural Volumetric Video Research},
  author={Xu, Zhen and Xie, Tao and Peng, Sida and Lin, Haotong and Shuai, Qing and Yu, Zhiyuan and He, Guangzhao and Sun, Jiaming and Bao, Hujun and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia 2023 Technical Communications},
  year={2023}
}

@article{xu20234k4d,
  title={4K4D: Real-Time 4D View Synthesis at 4K Resolution},
  author={Xu, Zhen and Peng, Sida and Lin, Haotong and He, Guangzhao and Sun, Jiaming and Shen, Yujun and Bao, Hujun and Zhou, Xiaowei},
  booktitle={arXiv preprint arXiv:2310.11448},
  year={2023}
}
```