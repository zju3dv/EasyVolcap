import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.console_utils import *


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        from timm.models.layers import trunc_normal_, DropPath
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


@REGRESSORS.register_module()
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 **kwargs,):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)
        self.size_pad = 32  # input size should be divisible by 32

    def _init_weights(self, m):
        from timm.models.layers import trunc_normal_, DropPath
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        # Remember input shapes, x: (B, S, C, H, W) or (B, C, H, W) or (C, H, W)
        sh = x.shape
        x = x.view(-1, *sh[-3:])  # (B, C, H, W)

        features = self.forward_features(x)
        # Restore input shapes
        features = [f.view(sh[:-3] + f.shape[-3:]) for f in features]
        return features


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def compute_depth_expectation(prob, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(prob * depth_values, 1)
    return depth


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()

        if kernel_size == 3:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, 3, padding=0, stride=1),
            )
        elif kernel_size == 1:
            self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1, padding=0, stride=1)

        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ConvBlock_double(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock_double, self).__init__()

        if kernel_size == 3:
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, 3, padding=0, stride=1),
            )
        elif kernel_size == 1:
            self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1, padding=0, stride=1)

        self.nonlin = nn.ELU(inplace=True)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 1, padding=0, stride=1)
        self.nonlin_2 = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        out = self.conv_2(out)
        out = self.nonlin_2(out)
        return out


class DecoderFeature(nn.Module):
    def __init__(self, feat_channels, num_ch_dec=[64, 64, 128, 256]):
        super(DecoderFeature, self).__init__()
        self.num_ch_dec = num_ch_dec
        self.feat_channels = feat_channels

        self.upconv_3_0 = ConvBlock(self.feat_channels[3], self.num_ch_dec[3], kernel_size=1)
        self.upconv_3_1 = ConvBlock_double(
            self.feat_channels[2] + self.num_ch_dec[3],
            self.num_ch_dec[3],
            kernel_size=1)

        self.upconv_2_0 = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2], kernel_size=3)
        self.upconv_2_1 = ConvBlock_double(
            self.feat_channels[1] + self.num_ch_dec[2],
            self.num_ch_dec[2],
            kernel_size=3)

        self.upconv_1_0 = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1], kernel_size=3)
        self.upconv_1_1 = ConvBlock_double(
            self.feat_channels[0] + self.num_ch_dec[1],
            self.num_ch_dec[1],
            kernel_size=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, ref_feature):
        x = ref_feature[3]

        x = self.upconv_3_0(x)
        x = torch.cat((self.upsample(x), ref_feature[2]), 1)
        x = self.upconv_3_1(x)

        x = self.upconv_2_0(x)
        x = torch.cat((self.upsample(x), ref_feature[1]), 1)
        x = self.upconv_2_1(x)

        x = self.upconv_1_0(x)
        x = torch.cat((self.upsample(x), ref_feature[0]), 1)
        x = self.upconv_1_1(x)
        return x


class UNet(nn.Module):
    def __init__(self, inp_ch=32, output_chal=1, down_sample_times=3, channel_mode='v0'):
        super(UNet, self).__init__()
        basic_block = ConvBnReLU
        num_depth = 128

        self.conv0 = basic_block(inp_ch, num_depth)
        if channel_mode == 'v0':
            channels = [num_depth, num_depth // 2, num_depth // 4, num_depth // 8, num_depth // 8]
        elif channel_mode == 'v1':
            channels = [num_depth, num_depth, num_depth, num_depth, num_depth, num_depth]
        self.down_sample_times = down_sample_times
        for i in range(down_sample_times):
            setattr(
                self, 'conv_%d' % i,
                nn.Sequential(
                    basic_block(channels[i], channels[i + 1], stride=2),
                    basic_block(channels[i + 1], channels[i + 1])
                )
            )
        for i in range(down_sample_times - 1, -1, -1):
            setattr(self, 'deconv_%d' % i,
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            channels[i + 1],
                            channels[i],
                            kernel_size=3,
                            padding=1,
                            output_padding=1,
                            stride=2,
                            bias=False),
                        nn.BatchNorm2d(channels[i]),
                        nn.ReLU(inplace=True)
                    )
                    )
            self.prob = nn.Conv2d(num_depth, output_chal, 1, stride=1, padding=0)

    def forward(self, x):
        features = {}
        conv0 = self.conv0(x)
        x = conv0
        features[0] = conv0
        for i in range(self.down_sample_times):
            x = getattr(self, 'conv_%d' % i)(x)
            features[i + 1] = x
        for i in range(self.down_sample_times - 1, -1, -1):
            x = features[i] + getattr(self, 'deconv_%d' % i)(x)
        x = self.prob(x)
        return x


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=pad,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


@REGRESSORS.register_module()
class HourglassDecoder(nn.Module):
    def __init__(self,
                 in_channels=[192, 384, 768, 1536],
                 decoder_channels=[128, 128, 256, 512],
                 depth_normalize=[0.3, 150.0],
                 num_samples=1,
                 non_negative=True,
                 ):
        super(HourglassDecoder, self).__init__()
        self.inchannels = in_channels
        self.decoder_channels = decoder_channels
        self.min_val = depth_normalize[0]
        self.max_val = depth_normalize[1]
        self.num_samples = num_samples
        self.non_negative = non_negative

        self.num_ch_dec = self.decoder_channels
        self.num_depth_regressor_anchor = 512
        self.feat_channels = self.inchannels
        unet_in_channel = self.num_ch_dec[1]
        unet_out_channel = 256

        self.decoder_mono = DecoderFeature(self.feat_channels, self.num_ch_dec)
        self.conv_out_2 = UNet(inp_ch=unet_in_channel,
                               output_chal=unet_out_channel + 1,
                               down_sample_times=3,
                               channel_mode='v0',
                               )

        self.depth_regressor_2 = nn.Sequential(
            nn.Conv2d(unet_out_channel,
                      self.num_depth_regressor_anchor,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.BatchNorm2d(self.num_depth_regressor_anchor),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.num_depth_regressor_anchor,
                self.num_depth_regressor_anchor,
                kernel_size=1,
            )
        )
        self.residual_channel = 16
        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(1 + 2 + unet_out_channel, self.residual_channel, 3, padding=1),
            nn.BatchNorm2d(self.residual_channel),
            nn.ReLU(),
            nn.Conv2d(self.residual_channel, self.residual_channel, 3, padding=1),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(self.residual_channel, self.residual_channel, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.residual_channel, 1, 1, padding=0),
            nn.ReLU() if self.non_negative else nn.Identity()
        )

    def get_bins(self, bins_num):
        depth_bins_vec = torch.linspace(math.log(self.min_val), math.log(self.max_val), bins_num, device='cuda')
        depth_bins_vec = torch.exp(depth_bins_vec)
        return depth_bins_vec

    def register_depth_expectation_anchor(self, bins_num):
        depth_bins_vec = self.get_bins(bins_num)
        depth_bins_vec = depth_bins_vec[None]
        self.register_buffer('depth_expectation_anchor', depth_bins_vec, persistent=False)

    def upsample(self, x, scale_factor=2):
        return F.interpolate(x, scale_factor=scale_factor, mode='nearest')

    def regress_depth_2(self, feature_map_d):
        prob = self.depth_regressor_2(feature_map_d).softmax(dim=1)
        if "depth_expectation_anchor" not in self._buffers:
            self.register_depth_expectation_anchor(self.num_depth_regressor_anchor)
        d = compute_depth_expectation(
            prob,
            self.depth_expectation_anchor
        ).unsqueeze(1)
        return d, prob

    def create_mesh_grid(self, height, width, batch, device="cuda", set_buffer=True):
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                               torch.arange(0, width, dtype=torch.float32, device=device)], indexing='ij')
        meshgrid = torch.stack((x, y))
        meshgrid = meshgrid.unsqueeze(0).repeat(batch, 1, 1, 1)
        return meshgrid

    def gather_discrete_topk(self, pdf):
        index = pdf.topk(k=self.num_samples, dim=-3).indices
        return pdf.gather(dim=-3, index=index)

    def forward(self, features_mono):
        '''
        trans_ref2src: list of transformation matrix from the reference view to source view. [B, 4, 4]
        inv_intrinsic_pool: list of inverse intrinsic matrix.
        features_mono: features of reference and source views. [[ref_f1, ref_f2, ref_f3, ref_f4],[src1_f1, src1_f2, src1_f3, src1_f4], ...].
        '''
        # Remember input shapes, x: (B, S, C, H, W) or (B, C, H, W) or (C, H, W)
        shs = [x.shape for x in features_mono]
        features_mono = [x.view(-1, *sh[-3:]) for x, sh in zip(features_mono, shs)]

        # Get encoder feature of the reference view
        ref_feat = features_mono
        feature_map_mono = self.decoder_mono(ref_feat)
        feature_map_mono_pred = self.conv_out_2(feature_map_mono)
        feature_map_d_2 = feature_map_mono_pred[:, :-1, :, :]

        depth_pred_2, probability = self.regress_depth_2(feature_map_d_2)  # (BS, 1, H//4, W//4), (BS, D, H//4, W//4)

        B, _, H, W = depth_pred_2.shape
        meshgrid = self.create_mesh_grid(H, W, B)

        scale_pred_mono = self.conv_up_2(torch.cat((depth_pred_2, meshgrid[:B, ...], feature_map_d_2), 1))  # (BS, 1, H, W)
        depth_pred_mono = self.upsample(depth_pred_2, scale_factor=4) + 1e-1 * scale_pred_mono  # (BS, 1, H, W)
        feats_pred_mono = feature_map_d_2  # (BS, C, H//4, W//4)
        alpha_pred_mono = self.upsample(self.gather_discrete_topk(probability), scale_factor=4)  # (BS, N, H, W)

        # Restore input shapes
        scale_pred_mono = scale_pred_mono.view(shs[0][:-3] + scale_pred_mono.shape[-3:])  # (B, S, 1, H, W)
        depth_pred_mono = depth_pred_mono.view(shs[0][:-3] + depth_pred_mono.shape[-3:])  # (B, S, 1, H, W)
        feats_pred_mono = feats_pred_mono.view(shs[0][:-3] + feats_pred_mono.shape[-3:])  # (B, S, C, H//4, W//4)
        alpha_pred_mono = alpha_pred_mono.view(shs[0][:-3] + alpha_pred_mono.shape[-3:])  # (B, S, N, H, W)

        return dotdict(depth=depth_pred_mono, scale=scale_pred_mono, feats=feats_pred_mono, alpha=alpha_pred_mono)
