```shell
# Training
# Normal training
evc-train -c configs/exps/enerfpp/enerfpp_actor1_4.yaml exp_name=enerfpp_actor1_4 runner_cfg.resume=False
# Special training with perceptual loss and extra source pool, remember to copy debug code from the notion site
evc-train -c configs/exps/enerfpp/enerfpp_static.yaml exp_name=enerfpp_static_wiperc runner_cfg.resume=False dataloader_cfg.dataset_cfg.patch_size=128,128 dataloader_cfg.dataset_cfg.extra_src_pool=2

# Rendering
# Render existing paths
evc-test -c configs/exps/enerfpp/enerfpp_static00.yaml,configs/specs/orbit.yaml,configs/specs/ibr.yaml exp_name=enerfpp_static val_dataloader_cfg.dataset_cfg.render_size=-1,-1 val_dataloader_cfg.dataset_cfg.save_interp_path=False val_dataloader_cfg.dataset_cfg.camera_path_intri=data/paths/iphone/static00/spiral_018_180/intri.yml val_dataloader_cfg.dataset_cfg.camera_path_extri=data/paths/iphone/static00/spiral_018_180/extri.yml val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1 val_dataloader_cfg.dataset_cfg.n_render_views=180 val_dataloader_cfg.dataset_cfg.interp_type=NONE val_dataloader_cfg.dataset_cfg.interp_cfg.smoothing_term=-1.0 model_cfg.sampler_cfg.cache_size=10 runner_cfg.visualizer_cfg.save_tag=static00_spiral_018_180_src36 runner_cfg.visualizer_cfg.types=RENDER,DEPTH,SRCINPS val_dataloader_cfg.dataset_cfg.view_sample=0,None,20
# Render original input size, with cropping
evc-test -c configs/exps/enerfpp/enerfpp_actor1_4_debug_tight.yaml,configs/specs/cubic.yaml,configs/specs/ibr.yaml exp_name=enerfpp_actor1_4_debug_tight val_dataloader_cfg.dataset_cfg.render_size=-1,-1 val_dataloader_cfg.dataset_cfg.frame_sample=0,120,1 val_dataloader_cfg.dataset_cfg.n_render_views=120 runner_cfg.visualizer_cfg.video_fps=30 runner_cfg.visualizer_cfg.save_tag=orig_size

# GUI rendering
evc-gui -c configs/exps/enerfpp/enerfpp_static00.yaml  exp_name=enerfpp_static val_dataloader_cfg.dataset_cfg.frame_sample=0,1,1 viewer_cfg.window_size=540,960 val_dataloader_cfg.dataset_cfg.view_sample=0,None,30
evc-gui -c configs/exps/enerfpp/enerfpp_complex_layout_conf1.yaml exp_name=enerfpp_static val_dataloader_cfg.dataset_cfg.frame_sample=0,100,1 viewer_cfg.window_size=480,640
```
