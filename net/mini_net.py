#! /usr/bin/env python
# coding: utf-8
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2 as cv

ToInt = lambda x: int(round(x))

import torch
import torch.nn.functional as F
from net.extract_feature import ExtractFeature
from net.make_layers import get_group_gn, group_norm, make_conv3x3, make_fc, conv_with_kaiming_uniform


def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class ROIPOOLING(torch.nn.Module):
    def __init__(self, resize=30):
        super(ROIPOOLING, self).__init__()
        # self.roipooling = torch.nn.AdaptiveAvgPool2d(resize)

    def forward(self, x, resize):
        # print(f'[ROIPOOLING][forward]  x: {type(x)}')
        rois = []
        for i, xi in enumerate(x):
            print(f'[ROIPOOLING][forward]  x-{i}: {xi.shape}')
            # roi = self.roipooling(xi)
            roi = torch.nn.functional.adaptive_avg_pool2d(xi, resize)
            rois.append(roi)
            print(f'[ROIPOOLING][forward]  roi: {roi.shape}')
        return torch.cat(rois)


class MiniNet(torch.nn.Module):
    def __init__(
        self, 
        in_channel: int, 
        roi_size: int, 
        strides=1, 
        batch_norm: bool=True,
        activation: str or None='sigmoid'
    ):
        super(MiniNet, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.roi_size = roi_size
        self.in_channel = in_channel
        self.strides = strides
        self.extract_feature = ExtractFeature()
        self.extract_scale = 2.0

        # self.roi = ROIPOOLING(self.roi_size)
        self.roi = ROIPOOLING()
        self.corr = xcorr_depthwise
        self.conv1 = make_conv3x3(self.in_channel, 1)
        self.conv2 = make_conv3x3(self.in_channel, 4)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, templete_features, templete_bboxes, search_features):
        # get sr-feature and templete-feture
        print(f'[MiniNet][forward]  templete_features-1: {templete_features.shape}')
        print(f'[MiniNet][forward]  templete_bboxes: {templete_bboxes}')
        print(f'[MiniNet][forward]  search_features-1: {search_features.shape}')
        
        # extract feature, make feature to sr-shape(C, 30, 30) and templete-shape(C, 15, 15)
        templete_features = self.extract_feature(templete_features, templete_bboxes, scale=1.0)
        search_features = self.extract_feature(search_features, templete_bboxes, scale=self.extract_scale)
        for i, tfi in enumerate(templete_features):
            print(f'[MiniNet][forward]  templete_features-{i}: {tfi.shape}')
        for i, sfi in enumerate(search_features):
            print(f'[MiniNet][forward]  search_features-{i}: {sfi.shape}')

        # roi-pooling
        # templete_features = self.roi(templete_features)
        templete_features = self.roi(templete_features, 15)
        print(f'[MiniNet][forward]  templete_features-2: {templete_features.shape}')
        # search_features = self.roi(search_features)
        search_features = self.roi(search_features, self.roi_size)
        print(f'[MiniNet][forward]  search_features-2: {search_features.shape}')
        
        # conv xcorr-feature
        corr = xcorr_depthwise(search_features, templete_features)
        print(f'[MiniNet][forward]  corr: {corr.shape}')

        # conv
        conv1 = self.conv1(corr)
        print(f'[MiniNet][forward]  conv1: {conv1.shape}')
        conv2 = self.conv2(corr)
        print(f'[MiniNet][forward]  conv2: {conv2.shape}')
        
        conf = self.sigmoid(conv1)
        xywh = self.sigmoid(conv2)

        return conf, xywh



if __name__ == '__main__':

    from utils.bbox.bbox import BBoxes
    
    batch_size = 1
    in_channel = 64
    roi_size = 30
    strides = 1
    bboxes = np.array([[10, 10, 50, 90], [40, 50, 80, 90]], dtype=np.float32)
    
    net = MiniNet(
        in_channel=in_channel,
        roi_size=roi_size,
        strides=strides
    )
    
    templete_bboxes = BBoxes(bboxes, mode='xyxy')
    search_features = torch.randn((batch_size, in_channel, 100, 100))
    templete_features = torch.randn((batch_size, in_channel, 100, 100))
    
    conf, xywh = net(templete_features, templete_bboxes, search_features)
    print(f'[main]  conf: {conf.shape}')
    print(f'[main]  xywh: {xywh.shape}')
    conf.sum().backward()