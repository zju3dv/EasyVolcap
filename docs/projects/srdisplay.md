# Spatial Reality Display

This is a brief record of developing with [Spatial Reality Display(elf-sr2)](https://www.sony.net/Products/Developer-Spatial-Reality-display/en/), namely integrate the real-time telepresence algorithm ([ENeRF](./enerf.md) and [4K4D](./realtime4dv.md) for now) and the elf-sr2 3D display with [Unity](https://www.sony.net/Products/Developer-Spatial-Reality-display/en/develop/Unity/Setup.html).

There is two parts of the experience: the **EasyVolcap** end python rendering server and the **Unity** end sr display client.

## **EasyVolcap** End

For the **EasyVolcap** end, we can use the common GUI rendeirng command, except changing the viewer type to `UnitySocketViewer`.

```shell
# 4K4D on the DNA-Rendering dataset
evc-gui -c configs/projects/realtime4dv/rendering/4k4d_0013_01.yaml val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1 viewer_cfg.type=UnitySocketViewer viewer_cfg.render_size=2160,3840 viewer_cfg.skip_exception=True viewer_cfg.autoplay=True viewer_cfg.scene_center_index=22

# ENeRF++ on the online telepresence system
evc-gui -c configs/exps/enerfpp/enerfpp_online_complex.yaml exp_name=enerfpp_static viewer_cfg.type=UnitySocketViewer viewer_cfg.render_size=480,640 viewer_cfg.skip_exception=True viewer_cfg.scene_center_index=2
```

## **Unity** End

For the **Unity** end, please refer to [Spatial Reality Display Developer Guide](https://www.sony.net/Products/Developer-Spatial-Reality-display/en/develop/Overview.html) for an overview of the Spatial Reality Display and the Unity SDK, and [Setup](https://www.sony.net/Products/Developer-Spatial-Reality-display/en/develop/Setup/SetupSRDisplay.html) inside for the setup of the display and the Unity SDK.

After the setup, clone the modified [Unity SDK](https://github.com/xbillowy/EasyVolcapUnityRenderingPlugin.git) and open the `EasyVolcapUnityRenderingPlugin` project with Unity. Then open the `EasyVolcapUnityRenderingPlugin/Assets/Scenes/SocketViewer.unity` scene, and run the scene. The scene will connect to the **EasyVolcap** end and render the scene.

```shell
git clone https://github.com/xbillowy/EasyVolcapUnityRenderingPlugin.git
```
