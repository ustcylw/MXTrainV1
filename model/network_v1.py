#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import cv2 as cv
import mxnet as mx


ToInt = lambda x: int(round(x))


class Input(mx.gluon.nn.HybridBlock):
    def __init__(self, multiplier=1.0, channels=[], activation='sigmoid', prefix=None, params=None):
        super(Input, self).__init__(prefix=prefix, params=params)
        self.multiplier = multiplier
        self.activation = activation
        self.channels = channels

        self.seq = mx.gluon.nn.HybridSequential()
        self.seq.add(
            mx.gluon.nn.Conv2D(ToInt(self.channels[0] * self.multiplier), 3, strides=(1, 1), padding=(1, 1),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0
            ),
            mx.gluon.nn.BatchNorm(),
            mx.gluon.nn.Activation(self.activation),
            mx.gluon.nn.Conv2D(ToInt(self.channels[1] * self.multiplier), 3, strides=2, padding=(1, 1),
                dilation=(1, 1), groups=1, layout='NCHW',
                activation=None, use_bias=True, weight_initializer=None,
                bias_initializer='zeros', in_channels=0
            ),
            mx.gluon.nn.BatchNorm(),
            mx.gluon.nn.Activation(self.activation)
        )

    def hybrid_forward(self, F, x):
        print(f'[Input] x: {x.shape}')
        y = self.seq(x)
        print(f'[Input] y: {y.shape}')
        return y


class Lemon(mx.gluon.nn.HybridBlock):
    def __init__(self, name='lemon', multiplier=1.0, activation='sigmoid', channels=[], att1=True, att2=True, prefix=None, params=None):
        super(Lemon, self).__init__(prefix=prefix, params=params)
        self.multiplier = multiplier
        self.activation = activation
        self.channels = channels if len(channels) > 0 else [512, 512, 512]
        self.att1 = att1
        self.att2 = att2
        # self.name = name

        self.seq = mx.gluon.nn.HybridSequential()
        self.seq.add(
            # [W, H] --> [W, H]
            mx.gluon.nn.Conv2D(ToInt(self.channels[0] * self.multiplier), 3, strides=(1, 1), padding=(1, 1),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0
            ),
            mx.gluon.nn.BatchNorm(),
            mx.gluon.nn.Activation(self.activation),
            # [W, H] --> [W/2, H/2]
            mx.gluon.nn.Conv2D(ToInt(self.channels[1] * self.multiplier), 3, strides=2, padding=(1, 1),
                dilation=(1, 1), groups=1, layout='NCHW',
                activation=None, use_bias=True, weight_initializer=None,
                bias_initializer='zeros', in_channels=0
            ),
            mx.gluon.nn.BatchNorm(),
            mx.gluon.nn.Activation(self.activation),
            # [W/2, H/2] --> [W/2, H/2]
            mx.gluon.nn.Conv2D(ToInt(self.channels[2] * self.multiplier), 3, strides=1, padding=(1, 1),
                dilation=(1, 1), groups=1, layout='NCHW',
                activation=None, use_bias=True, weight_initializer=None,
                bias_initializer='zeros', in_channels=0
            ),
            # mx.gluon.nn.BatchNorm(),
            # mx.gluon.nn.Activation(self.activation)
        )

        if self.att1:
            self.attention1 = mx.gluon.nn.HybridSequential()
            self.attention1.add(
                # [W/2, H/2] --> [W/2, H/2]
                mx.gluon.nn.Conv2D(ToInt(self.channels[0] * self.multiplier), 3, strides=1, padding=(1, 1),
                                   dilation=(1, 1), groups=1, layout='NCHW',
                                   activation=None, use_bias=True, weight_initializer=None,
                                   bias_initializer='zeros', in_channels=0
                                   ),
                mx.gluon.nn.BatchNorm(),
                mx.gluon.nn.Activation(self.activation),
                # [W/2, H/2] --> [W/4, H/4]
                mx.gluon.nn.Conv2D(ToInt(self.channels[1] * self.multiplier), 3, strides=2, padding=(1, 1),
                                   dilation=(1, 1), groups=1, layout='NCHW',
                                   activation=None, use_bias=True, weight_initializer=None,
                                   bias_initializer='zeros', in_channels=0
                                   ),
                mx.gluon.nn.BatchNorm(),
                mx.gluon.nn.Activation(self.activation),
                # [W/4, H/4] --> [W/2, H/2]
                mx.gluon.nn.Conv2D(ToInt(self.channels[2] * self.multiplier), 3, strides=2, padding=(1, 1),
                                   dilation=(1, 1), groups=1, layout='NCHW',
                                   activation=None, use_bias=True, weight_initializer=None,
                                   bias_initializer='zeros', in_channels=0
                                   ),
                mx.gluon.nn.BatchNorm(),
                mx.gluon.nn.Activation(self.activation),
                mx.gluon.nn.Conv2DTranspose(ToInt(self.channels[2] * self.multiplier), kernel_size=3, strides=2, padding=(1, 1),
                    output_padding=(1, 1), dilation=(1, 1), groups=1, layout='NCHW',
                    activation=None, use_bias=True, weight_initializer=None,
                    bias_initializer='zeros', in_channels=0
                ),
                # mx.gluon.nn.BatchNorm(),
                # mx.gluon.nn.Activation(self.activation),
            )
        else:
            self.attention1 = mx.gluon.nn.HybridSequential()

        if self.att2:
            self.attention2 = mx.gluon.nn.HybridSequential()
            self.attention2.add(
                # [W/2, H/2] --> [W/2, H/2]
                mx.gluon.nn.Conv2D(ToInt(self.channels[0] * self.multiplier), 3, strides=2, padding=(1, 1),
                                   dilation=(1, 1), groups=1, layout='NCHW',
                                   activation=None, use_bias=True, weight_initializer=None,
                                   bias_initializer='zeros', in_channels=0
                                   ),
                mx.gluon.nn.BatchNorm(),
                mx.gluon.nn.Activation(self.activation),
                # [W/2, H/2] --> [W/4, H/4]
                mx.gluon.nn.Conv2D(ToInt(self.channels[1] * self.multiplier), 3, strides=2, padding=(1, 1),
                                   dilation=(1, 1), groups=1, layout='NCHW',
                                   activation=None, use_bias=True, weight_initializer=None,
                                   bias_initializer='zeros', in_channels=0
                                   ),
                mx.gluon.nn.BatchNorm(),
                mx.gluon.nn.Activation(self.activation),
                # [W/4, H/4] --> [W/2, H/2]
                mx.gluon.nn.Conv2D(ToInt(self.channels[2] * self.multiplier), 3, strides=2, padding=(1, 1),
                                   dilation=(1, 1), groups=1, layout='NCHW',
                                   activation=None, use_bias=True, weight_initializer=None,
                                   bias_initializer='zeros', in_channels=0
                                   ),
                mx.gluon.nn.BatchNorm(),  #  ******
                mx.gluon.nn.Activation(self.activation),  # ******
                mx.gluon.nn.GlobalMaxPool2D(),
                # mx.gluon.nn.Flatten()
            )
        else:
            self.attention2 = mx.gluon.nn.HybridSequential()

        self.merge = mx.gluon.nn.HybridSequential()
        self.merge.add(
            mx.gluon.nn.BatchNorm(),
            mx.gluon.nn.Activation(self.activation)
        )

    def hybrid_forward(self, F, x):
        print(f'[Lemon] {self.name} x: {x.shape}')
        seq = self.seq(x)
        print(f'[Lemon] {self.name} seq: {seq.shape}')
        att1 = self.attention1(x)
        print(f'[Lemon] {self.name} att1: {att1.shape}')
        att2 = self.attention2(x)
        print(f'[Lemon] {self.name} att2: {att2.shape}')
        att = att1 * att2
        print(f'[Lemon] {self.name} att: {att.shape}')
        y = self.merge(seq + att)
        print(f'[Lemon] {self.name} y: {y.shape}')
        return y

class Output(mx.gluon.nn.HybridBlock):
    def __init__(self, multiplier=1.0, activation='sigmoid', in_channels=128, channels=[], keypoints=True, num_keypoints=1, prefix=None, params=None):
        super(Output, self).__init__(prefix=prefix, params=params)
        self.multiplier = multiplier
        self.activation = activation
        self.keypoints = keypoints
        self.num_keyppoints = num_keypoints
        self.in_channels = in_channels
        self.channels = channels

        self.landmarks = mx.gluon.nn.HybridSequential()
        self.landmarks.add(
            mx.gluon.nn.Conv2D(ToInt(self.channels[0] * self.multiplier), 3, strides=(1, 1), padding=(1, 1),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0
            ),
            mx.gluon.nn.BatchNorm(),
            mx.gluon.nn.Activation(self.activation),
            mx.gluon.nn.Conv2D(ToInt(self.channels[1] * self.multiplier), 3, strides=2, padding=(1, 1),
                dilation=(1, 1), groups=1, layout='NCHW',
                activation=None, use_bias=True, weight_initializer=None,
                bias_initializer='zeros', in_channels=0
            ),
            mx.gluon.nn.BatchNorm(),
            mx.gluon.nn.Activation(self.activation),
            mx.gluon.nn.Conv2D(136, 3, strides=(1, 1), padding=(0, 0),
                               dilation=(1, 1), groups=1, layout='NCHW',
                               activation=None, use_bias=True, weight_initializer=None,
                               bias_initializer='zeros', in_channels=0
                               ),
            # mx.gluon.nn.BatchNorm(),
            mx.gluon.nn.Activation(self.activation),
            mx.gluon.nn.GlobalMaxPool2D(),
            mx.gluon.nn.Flatten()
        )

        self.keypoints_seq = mx.gluon.nn.HybridSequential()
        self.keypoints_seq.add(
            mx.gluon.nn.Conv2D(ToInt(self.channels[0] * self.multiplier), 3, strides=(1, 1), padding=(1, 1),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0
            ),
            mx.gluon.nn.BatchNorm(),
            mx.gluon.nn.Activation(self.activation),
            mx.gluon.nn.Conv2D(ToInt(self.channels[1] * self.multiplier), 3, strides=2, padding=(1, 1),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0
            ),
            mx.gluon.nn.BatchNorm(),
            mx.gluon.nn.Activation(self.activation),
            mx.gluon.nn.Conv2D(self.num_keyppoints * 2, 3, strides=1, padding=(1, 1),
                dilation=(1, 1), groups=1, layout='NCHW',
                activation=None, use_bias=True, weight_initializer=None,
                bias_initializer='zeros', in_channels=0
            ),
            # mx.gluon.nn.BatchNorm(),
            mx.gluon.nn.Activation(self.activation),
            mx.gluon.nn.GlobalMaxPool2D(),
            mx.gluon.nn.Flatten()
        )

    def hybrid_forward(self, F, x):
        print(f'[Output] x: {x.shape}')
        landmarks = self.landmarks(x)
        print(f'[Output] landmark: {landmarks.shape}')
        keypoints = self.keypoints_seq(x)
        print(f'[Output] keypoints: {keypoints.shape}')
        return landmarks, keypoints


class IQ(mx.gluon.nn.HybridBlock):
    def __init__(self, multiplier=1.0, channels=[], activation='sigmoid', att1=True, att2=True, keypoints=True, num_keyppoints=1, prefix=None, params=None):
        super(IQ, self).__init__(prefix=prefix, params=params)
        self.multiplier = multiplier
        self.activation = activation
        self.channels = channels
        self.att1 = att1
        self.att2 = att2
        self.keypoints = keypoints
        self.num_keypoints = num_keyppoints

        self.input = Input(multiplier=self.multiplier, channels=self.channels[0], activation=self.activation)
        self.lemon1 = Lemon(name='lemon1', multiplier=self.multiplier, channels=self.channels[1], activation=self.activation, att1=self.att1, att2=self.att2, )
        self.lemon2 = Lemon(name='lemon2', multiplier=self.multiplier, channels=self.channels[1], activation=self.activation, att1=self.att1, att2=self.att2, )
        self.lemon3 = Lemon(name='lemon3', multiplier=self.multiplier, channels=self.channels[1], activation=self.activation, att1=self.att1, att2=self.att2, )
        self.output = Output(multiplier=self.multiplier, channels=self.channels[2], keypoints=self.keypoints, num_keypoints=self.num_keypoints, activation=self.activation)

    def hybrid_forward(self, F, x):
        #  128 X 128 --> 64 X 64
        input = self.input(x)
        #  64 X 64 --> 32 X 32
        lemon1 = self.lemon1(input)
        #  32 X 32 --> 16 X 16
        lemon2= self.lemon2(lemon1)
        #  16 X 16 --> 8 X 8
        lemon3 = self.lemon3(lemon2)
        #  8 X 8 --> 4 X 4 --> (136, 2*num_keypints)
        output = self.output(lemon3)
        return output




if __name__ == '__main__':

    net = IQ(
        multiplier=1.0,
        channels=[[128, 256], [512, 512, 512], [256, 136]],
        activation='sigmoid',
        att1=True,
        att2=True,
        keypoints=True,
        num_keyppoints=1
    )
    net.initialize()

    x = mx.random.uniform(shape=(2, 3, 128, 128))

    y = net(x)
    print(f'[main] y: {len(y)}  {y[0].shape}  {y[1].shape}')

