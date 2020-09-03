# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ
"""MobileNet and MobileNetV2, implemented in Gluon."""
__all__ = ['MobileNetV2', 'mobilenet_v2_1_0',
           'mobilenet_v2_0_75', 'mobilenet_v2_0_5',
           'mobilenet_v2_0_25', 'get_mobilenet_v2']

__modify__ = 'dwSun'
__modified_date__ = '18/04/18'

import os
import mxnet as mx
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet import base


# Helpers
class RELU6(nn.HybridBlock):
    """Relu6 used in MobileNetV2."""

    def __init__(self, **kwargs):
        super(RELU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")


# pylint: disable= too-many-arguments
def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(nn.BatchNorm(scale=True))
    if active:
        out.add(RELU6() if relu6 else nn.Activation('relu'))


def _add_conv_dw(out, dw_channels, channels, stride, relu6=False):
    _add_conv(out, channels=dw_channels, kernel=3, stride=stride,
              pad=1, num_group=dw_channels, relu6=relu6)
    _add_conv(out, channels=channels, relu6=relu6)


class LinearBottleneck(nn.HybridBlock):
    def __init__(self, in_channels, channels, t, stride, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        with self.name_scope():
            self.out = nn.HybridSequential()

            _add_conv(self.out, in_channels * t, relu6=False)
            _add_conv(self.out, in_channels * t, kernel=3, stride=stride,
                      pad=1, num_group=in_channels * t, relu6=False)
            _add_conv(self.out, channels, active=False, relu6=False)

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


class MobileNetV2(nn.HybridBlock):
    def __init__(self, multiplier=1.0, classes=1000, **kwargs):
        super(MobileNetV2, self).__init__(**kwargs)
        with self.name_scope():
            self.base = nn.HybridSequential(prefix='base_')
            with self.base.name_scope():
                _add_conv(self.base, int(32 * multiplier), kernel=3,
                          stride=1, pad=1, relu6=False)

                # in_channels_group = [int(x * multiplier) for x in [32] + [16] + [24] * 2
                #                      + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
                # channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [32] * 3
                #                   + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
                # ts = [1] + [6] * 16
                # strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3

                in_channels_group = [int(x * multiplier) for x in [32] + [16] + [24] * 2
                                     + [32] * 3 + [64] * 3]  # 4 + [96] * 3 + [160] * 3]
                channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [32] * 3
                                  + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
                ts = [1] + [6] * 16
                strides = [1, 2] * 2 + [1, 1, 2] + [1] + [2] + [1]  # 6 + [2] + [1] * 3

                for in_c, c, t, s in zip(in_channels_group, channels_group, ts, strides):
                    self.base.add(LinearBottleneck(in_channels=in_c, channels=c,
                                                       t=t, stride=s))

                # last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
                # _add_conv(self.features, last_channels, relu6=True)

            self.b1 = nn.HybridSequential(prefix='b1_')
            with self.b1.name_scope():
                self.b1.add(nn.Conv2D(32, 3, 2, padding=1))
                self.b1.add(nn.BatchNorm(scale=True))
                self.b1.add(mx.gluon.nn.Activation('relu'))
                self.b1.add(nn.Conv2D(classes, 1, 1, use_bias=False))
                self.b1.add(nn.Activation('sigmoid'))
                self.b1.add(nn.GlobalAvgPool2D())
                self.b1.add(nn.Flatten())
            # self.b2 = nn.HybridSequential(prefix='b2_')
            # with self.b2.name_scope():
            #     self.b2.add(nn.Conv2D(256, 3, 1, padding=1))
            #     self.b2.add(nn.BatchNorm(scale=True))
            #     self.b1.add(mx.gluon.nn.Activation('relu'))
            #     self.b2.add(nn.Conv2D(classes, 1, 1, use_bias=False))
            #     self.b2.add(nn.Activation('sigmoid'))
            #     self.b2.add(nn.GlobalAvgPool2D())
            #     self.b2.add(nn.Flatten())
            # self.b3 = nn.HybridSequential(prefix='b3_')
            # with self.b3.name_scope():
            #     self.b3.add(nn.Conv2D(256, 3, 2, padding=1))
            #     self.b3.add(nn.BatchNorm(scale=True))
            #     self.b3.add(RELU6())
            #     self.b3.add(nn.Conv2D(128, 3, 1, padding=1))
            #     self.b3.add(nn.BatchNorm(scale=True))
            #     self.b3.add(RELU6())
            #     self.b3.add(nn.GlobalAvgPool2D())
            #     self.b3.add(nn.Conv2D(10, 1, 1, use_bias=False))
            #     self.b3.add(nn.Activation('sigmoid'))
            #     self.b3.add(nn.Flatten())
            # self.b4 = nn.HybridSequential(prefix='b4_')
            # with self.b4.name_scope():
            #     self.b4.add(nn.Conv2D(256, 3, 2, padding=1))
            #     self.b4.add(nn.BatchNorm(scale=True))
            #     self.b4.add(RELU6())
            #     self.b4.add(nn.Conv2D(128, 3, 1, padding=1))
            #     self.b4.add(nn.BatchNorm(scale=True))
            #     self.b4.add(RELU6())
            #     self.b4.add(nn.GlobalAvgPool2D())
            #     self.b4.add(nn.Conv2D(10, 1, 1, use_bias=False))
            #     self.b4.add(nn.Activation('sigmoid'))
            #     self.b4.add(nn.Flatten())
            # self.b5 = nn.HybridSequential(prefix='b5_')
            # with self.b5.name_scope():
            #     self.b5.add(nn.Conv2D(256, 3, 2, padding=1))
            #     self.b5.add(nn.BatchNorm(scale=True))
            #     self.b5.add(RELU6())
            #     self.b5.add(nn.Conv2D(128, 3, 1, padding=1))
            #     self.b5.add(nn.BatchNorm(scale=True))
            #     self.b5.add(RELU6())
            #     self.b5.add(nn.GlobalAvgPool2D())
            #     self.b5.add(nn.Conv2D(18, 1, 1, use_bias=False))
            #     self.b5.add(nn.Activation('sigmoid'))
            #     self.b5.add(nn.Flatten())
            # self.b6 = nn.HybridSequential(prefix='b6_')
            # with self.b6.name_scope():
            #     self.b6.add(nn.Conv2D(256, 3, 2, padding=1))
            #     self.b6.add(nn.BatchNorm(scale=True))
            #     self.b6.add(RELU6())
            #     self.b6.add(nn.Conv2D(128, 3, 1, padding=1))
            #     self.b6.add(nn.BatchNorm(scale=True))
            #     self.b6.add(RELU6())
            #     self.b6.add(nn.GlobalAvgPool2D())
            #     self.b6.add(nn.Conv2D(12, 1, 1, use_bias=False))
            #     self.b6.add(nn.Activation('sigmoid'))
            #     self.b6.add(nn.Flatten())
            # self.b7 = nn.HybridSequential(prefix='b7_')
            # with self.b7.name_scope():
            #     self.b7.add(nn.Conv2D(256, 3, 2, padding=1))
            #     self.b7.add(nn.BatchNorm(scale=True))
            #     self.b7.add(RELU6())
            #     self.b7.add(nn.Conv2D(128, 3, 1, padding=1))
            #     self.b7.add(nn.BatchNorm(scale=True))
            #     self.b7.add(RELU6())
            #     self.b7.add(nn.GlobalAvgPool2D())
            #     self.b7.add(nn.Conv2D(12, 1, 1, use_bias=False))
            #     self.b7.add(nn.Activation('sigmoid'))
            #     self.b7.add(nn.Flatten())
            # self.b8 = nn.HybridSequential(prefix='b8_')
            # with self.b8.name_scope():
            #     self.b8.add(nn.Conv2D(256, 3, 2, padding=1))
            #     self.b8.add(nn.BatchNorm(scale=True))
            #     self.b8.add(RELU6())
            #     self.b8.add(nn.Conv2D(128, 3, 1, padding=1))
            #     self.b8.add(nn.BatchNorm(scale=True))
            #     self.b8.add(RELU6())
            #     self.b8.add(nn.GlobalAvgPool2D())
            #     self.b8.add(nn.Conv2D(40, 1, 1, use_bias=False))
            #     self.b8.add(nn.Activation('sigmoid'))
            #     self.b8.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        base = self.base(x)
        # print('[===]  base: ', base.shape)
        b1 = self.b1(base)
        # print('[===]  b1: ', b1.shape)
        # b2 = self.b2(base)
        # print('[===]  b2: ', b2.shape)
        # b3 = self.b3(base)
        # # print('[===]  b3: ', b3.shape)
        # b4 = self.b4(base)
        # # print('[===]  b4: ', b4.shape)
        # b5 = self.b5(base)
        # # print('[===]  b5: ', b5.shape)
        # b6 = self.b6(base)
        # # print('[===]  b6: ', b6.shape)
        # b7 = self.b7(base)
        # # print('[===]  b7: ', b7.shape)
        # b8 = self.b8(base)
        # # print('[===]  b8: ', b8.shape)
        return b1  # , b2 , b3, b4, b5, b6, b7, b8


def get_mobilenet_v2(multiplier, pretrained=False, ctx=cpu(),
                     root=os.path.join(base.data_dir(), 'models'), **kwargs):
    net = MobileNetV2(multiplier, **kwargs)
    return net


def mobilenet_v2_1_0(**kwargs):
    return get_mobilenet_v2(1.0, **kwargs)


def mobilenet_v2_0_75(**kwargs):
    return get_mobilenet_v2(0.75, **kwargs)


def mobilenet_v2_0_5(**kwargs):
    return get_mobilenet_v2(0.5, **kwargs)


def mobilenet_v2_0_25(**kwargs):
    return get_mobilenet_v2(0.25, **kwargs)


if __name__ == '__main__':

    import numpy as np

    data_shape = (3, 128, 128)
    # data_shape = (3, 64, 64)
    W, H = data_shape[2], data_shape[2]
    # data = mx.nd.array(np.linspace(0, W*H*3-1, W*H*3).reshape(1, 3, W, H))
    data = mx.nd.random.uniform(shape=(2, 3, W, H), dtype=np.float32)

    net = get_mobilenet_v2(1.0, classes=136)
    net.initialize(init=mx.init.Normal(sigma=0.01), force_reinit=True)
    W, H = data_shape[2], data_shape[2]
    x = mx.random.uniform(shape=(2, 3, W, H))
    net.summary(x)

    y = net(data)
    print('y: ', y[0].shape, y[1].shape)
    net.hybridize()
    y = net.forward(x)

    prefix = f'../checkpoints/FM/001/ok/init-{W}-{H}'
    net.export(prefix, epoch=0)

    sym, _, _ = mx.model.load_checkpoint(prefix=prefix, epoch=0)
    digraph = mx.viz.plot_network(sym, title=prefix+f'-{W}-{H}-v3', save_format='pdf', shape={'data':(1, 3, W, H)})
    digraph.view()
