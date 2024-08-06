如何给一段sequence（evc format）用RC做dense重建

1. 部分的手工处理：需要import 第一帧的colmap格式进 RC，然后再export alignment metadata(xmp) 到data/xmps (之后每帧的rc 重建都会使用这个pose)
2. 跑 process.py
3. 跑 post_process.py
4. 跑 https://github.com/dendenxu/easyvolcap/blob/626f3c2fe272237d009f6fd05855827d0f5be63a/scripts/reality_capture/align_mesh_to_evc.py，因为即使提前import colmap pose into RC, RC还是改了坐标系（可能是调换了坐标轴），所以需要统一的重新align一下
