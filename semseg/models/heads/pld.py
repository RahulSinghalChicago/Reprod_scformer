from einops import rearrange
from torch.nn import *
from mmcv.cnn import build_activation_layer, build_norm_layer
#from timm.models.layers import DropPath
from einops.layers.torch import Rearrange
import numpy as np
import torch
from torch.nn import Module, ModuleList, Upsample
from mmcv.cnn import ConvModule
from torch.nn import Sequential, Conv2d, UpsamplingBilinear2d
import torch.nn as nn

class PLDHead(Module):

    def __init__(self, dims: list, dim: int = 256, class_num: int = 2):
        
        super(PLDHead, self).__init__()
        self.class_num = class_num
        self.layers = ModuleList([Sequential(Conv2d(dims[i], dim, 1), Upsample(scale_factor=2 ** i)) for i in range(len(dims))])

        self.conv_fuse = ConvModule(in_channels=dim * 4, out_channels=dim, kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        
        self.conv_fuse_12 = ConvModule(in_channels=dim * 2,out_channels=dim,kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.conv1 = ConvModule(in_channels=dim,out_channels=self.class_num,kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.conv2 = ConvModule(in_channels=dim,out_channels=self.class_num,kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))

        self.Conv2d = Conv2d(self.class_num * 2, self.class_num, 1)

    def forward(self, features):
        fuse = []
        for feature, layer in zip(features, self.layers):
            fuse.append(layer(feature))
            
        fused_12 = torch.cat([fuse[0], fuse[1]], dim=1)  # ([1, 1536, 128, 128])
        fused_12 = self.conv_fuse_12(fused_12)  # 2, 768, 128, 128
        fused_12 = self.conv1(fused_12)

        fuse = torch.cat(fuse, dim=1)  # [1, 3072, 128, 128]
        fuse = self.conv_fuse(fuse)  # [1, 768, 128, 128]
        fuse = self.conv2(fuse)  # [1, 768, 128, 128]
        fused = torch.cat([fuse, fused_12], dim=1)  # [1, 1536, 128, 128]
        fused = self.Conv2d(fused)

        return fused