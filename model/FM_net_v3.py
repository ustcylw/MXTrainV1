#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import cv2 as cv
import mxnet as mx


ToInt = lambda x: int(round(x))


class Conv(mx.gluon.nn.HybridBlock):
    def __init__(self, channel: int, kernel=3, strides=1, padding=0, batch_norm: bool=True, activation: str or None='sigmoid', num_groups: int=1, prefix=None, params=None):
        super(Conv, self).__init__(prefix=prefix, params=params)
        self.batch_norm = batch_norm
        self.activation = activation
        self.channel = channel
        self.num_groups = num_groups
        self.kernel = kernel
        self.strides = strides
        self.padding = padding

        with self.name_scope():
            self.seq = mx.gluon.nn.HybridSequential()
            self.seq.add(
                mx.gluon.nn.Conv2D(self.channel, kernel_size=self.kernel, strides=self.strides, padding=self.padding,
                     dilation=(1, 1), groups=self.num_groups, layout='NCHW',
                     activation=None, use_bias=True, weight_initializer=None,
                     bias_initializer='zeros', in_channels=0
                ),
            )
            if self.batch_norm:
                self.seq.add(
                    mx.gluon.nn.BatchNorm(),
                )
            if self.activation is not None:
                self.seq.add(
                    mx.gluon.nn.Activation(self.activation),
                )

    def hybrid_forward(self, F, x):
        # print(f'[Conv] x: {x.shape if not isinstance(x, mx.symbol.Symbol) else x}')
        y = self.seq(x)
        # print(f'[Conv] y: {y.shape if not isinstance(y, mx.symbol.Symbol) else y}')
        return y


class ConvDW(mx.gluon.nn.HybridBlock):
    def __init__(self, in_channel: int, channel: int, strides=1, batch_norm: bool=True, activation: str or None='sigmoid', prefix=None, params=None):
        super(ConvDW, self).__init__(prefix=prefix, params=params)
        self.batch_norm = batch_norm
        self.activation = activation
        self.channel = channel
        self.in_channel = in_channel
        self.strides = strides
        self.use_shortcut = strides == 1 and in_channel == channel

        with self.name_scope():
            self.conv1 = Conv(kernel=1, strides=1, padding=0, channel=self.channel, batch_norm=True, activation=self.activation, num_groups=1)
            self.conv2 = Conv(kernel=3, strides=self.strides, padding=1, channel=self.channel, batch_norm=True, activation=self.activation, num_groups=self.channel)
            if self.use_shortcut:
                self.conv3 = Conv(kernel=1, strides=1, padding=0, channel=self.channel, batch_norm=False, activation=None, num_groups=1)
                if self.batch_norm:
                    self.bn = mx.gluon.nn.BatchNorm()
                if self.activation is not None:
                    self.act = mx.gluon.nn.Activation(self.activation)
            else:
                self.conv3 = Conv(kernel=1, strides=1, padding=0, channel=self.channel, batch_norm=self.batch_norm, activation=self.activation, num_groups=1)

    def hybrid_forward(self, F, x):
        print(f'[ConvDW] [{self.strides}]  x: {x.shape if not isinstance(x, mx.symbol.Symbol) else x}  {type(self.conv1)}  {type(self.conv2)}  {type(self.conv3)}')
        conv1 = self.conv1(x)
        # print(f'[ConvDW] conv1: {conv1.shape if not isinstance(conv1, mx.symbol.Symbol) else conv1}')
        conv2 = self.conv2(conv1)
        # print(f'[ConvDW] conv2: {conv2.shape if not isinstance(conv2, mx.symbol.Symbol) else conv2}')
        conv3 = self.conv3(conv2)
        # print(f'[ConvDW] conv3: {conv3.shape if not isinstance(conv3, mx.symbol.Symbol) else conv3}')
        if self.use_shortcut:
            y = conv3 + x
            # y = F.elemwise_add(conv1, conv3)
            # print(f'[ConvDW] shortcut: {y.shape if not isinstance(y, mx.symbol.Symbol) else y}')
            # y = F.BatchNorm(y)
            if self.batch_norm:
                y = self.bn(y)
                # print(f'[ConvDW] batch-norm: {y.shape if not isinstance(y, mx.symbol.Symbol) else y}')
            if self.activation is not None:
                # y = F.Activation(data=y, act_type=self.activation)
                y = self.act(y)
                # print(f'[ConvDW] {self.activation}: {y.shape if not isinstance(y, mx.symbol.Symbol) else y}')
        else:
            y = conv3
        print(f'[ConvDW]  [{self.strides}]  {self.name}  y: {y.shape if not isinstance(y, mx.symbol.Symbol) else y}')
        return y


class Input(mx.gluon.nn.HybridBlock):
    def __init__(self, in_channel: int, channels: list, multiplier=1.0, activation='sigmoid', prefix=None, params=None):
        super(Input, self).__init__(prefix=prefix, params=params)
        self.multiplier = multiplier
        self.activation = activation
        self.channels = channels
        self.in_channel = in_channel

        with self.name_scope():
            self.seq = mx.gluon.nn.HybridSequential()
            self.seq.add(
                mx.gluon.nn.Conv2D(self.channels[0], 3, strides=(1, 1), padding=(1, 1),
                     dilation=(1, 1), groups=1, layout='NCHW',
                     activation=None, use_bias=True, weight_initializer=None,
                     bias_initializer='zeros', in_channels=0
                ),
                mx.gluon.nn.BatchNorm(),
                mx.gluon.nn.Activation(self.activation),
                ConvDW(in_channel=self.channels[0], channel=ToInt(self.channels[1] * self.multiplier), strides=2, batch_norm=True, activation='sigmoid')
            )

    def hybrid_forward(self, F, x):
        print(f'[Input] x: {x.shape if not isinstance(x, mx.symbol.Symbol) else x}')
        y = self.seq(x)
        print(f'[Input] y: {y.shape if not isinstance(y, mx.symbol.Symbol) else y}')
        return y


class Lemon(mx.gluon.nn.HybridBlock):
    def __init__(self, in_channel: int, channels: list, strides: list=[2, 2, 2], name='lemon', multiplier=1.0, activation='sigmoid', att1=True, att2=True, prefix=None, params=None):
        super(Lemon, self).__init__(prefix=prefix, params=params)
        self.multiplier = multiplier
        self.activation = activation
        self.in_channel = in_channel
        self.strides = strides
        self.channels = channels if len(channels) > 0 else [512, 512, 512]
        self.att1 = att1
        self.att2 = att2
        # self.name = name
        print(f'[Lemon][init]  strides: {self.strides}')

        with self.name_scope():
            self.seq = mx.gluon.nn.HybridSequential()
            self.seq.add(
                # [W, H] --> [W, H]
                ConvDW(in_channel=ToInt(self.in_channel * self.multiplier), channel=ToInt(self.channels[0] * self.multiplier), strides=self.strides[0][0], batch_norm=True, activation='sigmoid'),
                # [W, H] --> [W/2, H/2]
                ConvDW(in_channel=ToInt(self.channels[0] * self.multiplier), channel=ToInt(self.channels[1] * self.multiplier), strides=self.strides[0][1], batch_norm=True, activation='sigmoid'),
                # [W/2, H/2] --> [W/2, H/2]
                mx.gluon.nn.Conv2D(ToInt(self.channels[2] * self.multiplier), 3, strides=self.strides[0][2], padding=(1, 1),
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
                    ConvDW(in_channel=ToInt(self.in_channel * self.multiplier),
                           channel=ToInt(self.channels[0] * self.multiplier), strides=self.strides[1][0], batch_norm=True,
                           activation='sigmoid'),
                    # [W/2, H/2] --> [W/4, H/4]
                    ConvDW(in_channel=ToInt(self.channels[0] * self.multiplier),
                           channel=ToInt(self.channels[1] * self.multiplier), strides=self.strides[1][1], batch_norm=True,
                           activation='sigmoid'),
                    # [W/4, H/4] --> [W/2, H/2]
                    ConvDW(in_channel=ToInt(self.channels[1] * self.multiplier),
                           channel=ToInt(self.channels[2] * self.multiplier), strides=self.strides[1][2], batch_norm=True,
                           activation='sigmoid'),
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
                self.attention1.add(
                    ConvDW(in_channel=ToInt(self.in_channel * self.multiplier),
                           channel=ToInt(self.channels[2] * self.multiplier), strides=2, batch_norm=True,
                           activation='sigmoid'),
                )

            if self.att2:
                self.attention2 = mx.gluon.nn.HybridSequential()
                print(f'[===]  channels: {self.channels}  {self.multiplier}, {self.strides}')
                self.attention2.add(
                    # [W/2, H/2] --> [W/2, H/2]
                    ConvDW(in_channel=ToInt(self.in_channel * self.multiplier),
                           channel=ToInt(self.channels[0] * self.multiplier), strides=self.strides[2][0], batch_norm=True,
                           activation='sigmoid'),
                    # [W/2, H/2] --> [W/4, H/4]
                    ConvDW(in_channel=ToInt(self.channels[0] * self.multiplier),
                           channel=ToInt(self.channels[1] * self.multiplier), strides=self.strides[2][1], batch_norm=True,
                           activation='sigmoid'),
                    # [W/4, H/4] --> [W/2, H/2]
                    ConvDW(in_channel=ToInt(self.channels[1] * self.multiplier),
                           channel=ToInt(self.channels[2] * self.multiplier), strides=self.strides[2][2], batch_norm=True,
                           activation='sigmoid'),
                    mx.gluon.nn.GlobalMaxPool2D(),
                    # mx.gluon.nn.Flatten()
                )
            # else:
            #     self.attention2 = mx.gluon.nn.HybridSequential()
            #     self.attention2.add(
            #         # [W/4, H/4] --> [W/2, H/2]
            #         mx.gluon.nn.Conv2D(ToInt(self.channels[2] * self.multiplier), 3, strides=2, padding=(1, 1),
            #                            dilation=(1, 1), groups=1, layout='NCHW',
            #                            activation=None, use_bias=True, weight_initializer=None,
            #                            bias_initializer='zeros', in_channels=0
            #                            ),
            #         mx.gluon.nn.BatchNorm(),  # ******
            #         mx.gluon.nn.Activation(self.activation),  # ******
            #     )

            self.merge = mx.gluon.nn.HybridSequential()
            self.merge.add(
                mx.gluon.nn.BatchNorm(),
                mx.gluon.nn.Activation(self.activation)
            )

    def hybrid_forward(self, F, x):
        print(f'[Lemon] {self.name} x: {x.shape if not isinstance(x, mx.symbol.Symbol) else x}')
        seq = self.seq(x)
        print(f'[Lemon] {self.name} seq: {seq.shape if not isinstance(seq, mx.symbol.Symbol) else seq}')
        att1 = self.attention1(x)
        print(f'[Lemon] {self.name} att1: {type(att1)} {att1.shape if not isinstance(att1, mx.symbol.Symbol) else att1}')
        if self.att2:
            att2 = self.attention2(x)
            print(f'[Lemon] {self.name} att2: {type(att2)} {att2.shape if not isinstance(att2, mx.symbol.Symbol) else att2}')
            if isinstance(att2, mx.symbol.Symbol):
                att2 = mx.symbol.broadcast_like(att2, att1)
                print(f'[Lemon] {self.name} broadcast-att2: {att2.shape if not isinstance(att2, mx.symbol.Symbol) else att2}')
            att = att1 * att2
            print(f'[Lemon] {self.name} att: {att.shape if not isinstance(att, mx.symbol.Symbol) else att}')
        else:
            att = att1
        y = self.merge(seq + att)
        print(f'[Lemon] {self.name} y: {y.shape if not isinstance(y, mx.symbol.Symbol) else y}')
        return y

class Output(mx.gluon.nn.HybridBlock):
    def __init__(self, multiplier=1.0, activation='sigmoid', in_channels=128, channels=[], strides=[], keypoints=True, num_keypoints=1, prefix=None, params=None):
        super(Output, self).__init__(prefix=prefix, params=params)
        self.multiplier = multiplier
        self.activation = activation
        self.keypoints = keypoints
        self.num_keyppoints = num_keypoints
        self.in_channels = in_channels
        self.channels = channels
        self.strides = strides

        with self.name_scope():
            self.landmarks = mx.gluon.nn.HybridSequential()
            self.landmarks.add(
                ConvDW(in_channel=ToInt(self.in_channels * self.multiplier),
                       channel=ToInt(self.channels[0] * self.multiplier), strides=self.strides[0], batch_norm=True,
                       activation='sigmoid'),
                ConvDW(in_channel=ToInt(self.channels[0] * self.multiplier),
                       channel=ToInt(self.channels[1] * self.multiplier), strides=self.strides[1], batch_norm=True,
                       activation='sigmoid'),
                mx.gluon.nn.Conv2D(channels=136, kernel_size=3, strides=self.strides[2], padding=(1, 1),
                                   dilation=(1, 1), groups=1, layout='NCHW',
                                   activation=None, use_bias=True, weight_initializer=None,
                                   bias_initializer='zeros', in_channels=0
                                   ),
                # mx.gluon.nn.BatchNorm(),
                mx.gluon.nn.Activation(self.activation),
                mx.gluon.nn.GlobalMaxPool2D(),
                mx.gluon.nn.Flatten()
            )

            if self.keypoints:
                self.keypoints_seq = mx.gluon.nn.HybridSequential()
                self.keypoints_seq.add(
                    ConvDW(in_channel=ToInt(self.in_channels * self.multiplier),
                           channel=ToInt(self.channels[0] * self.multiplier), strides=self.strides[0], batch_norm=True,
                           activation='sigmoid'),
                    ConvDW(in_channel=ToInt(self.channels[0] * self.multiplier),
                           channel=ToInt(self.channels[1] * self.multiplier), strides=self.strides[1], batch_norm=True,
                           activation='sigmoid'),
                    mx.gluon.nn.Conv2D(self.num_keyppoints * 2, 3, strides=self.strides[2], padding=(1, 1),
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
        print(f'[Output] x: {x.shape if not isinstance(x, mx.symbol.Symbol) else x}')
        landmarks = self.landmarks(x)
        print(f'[Output] landmark: {landmarks.shape if not isinstance(landmarks, mx.symbol.Symbol) else landmarks}')
        if self.keypoints:
            keypoints = self.keypoints_seq(x)
            print(f'[Output] keypoints: {keypoints.shape if not isinstance(keypoints, mx.symbol.Symbol) else keypoints}')
            return landmarks, keypoints
        else:
            return landmarks
        #     keypoints = None
        # return landmarks, keypoints


class IQ(mx.gluon.nn.HybridBlock):
    def __init__(self, multiplier=1.0, in_channel=3, channels=[], strides=[], num_lemons=3, activation='sigmoid', att1=True, att2=True, keypoints=True, num_keyppoints=1, prefix=None, params=None):
        super(IQ, self).__init__(prefix=prefix, params=params)
        self.multiplier = multiplier
        self.activation = activation
        self.channels = channels
        self.att1 = att1
        self.att2 = att2
        self.keypoints = keypoints
        self.num_keypoints = num_keyppoints
        self.strides = strides
        self.num_lemons = num_lemons
        print(f'[]  strides: {self.strides}')

        with self.name_scope():
            self.input = Input(multiplier=self.multiplier, in_channel=in_channel, channels=self.channels[0], activation=self.activation)
            self.lemons = mx.gluon.nn.HybridSequential()
            with self.lemons.name_scope():
                print(f'[IQ][init]  strides: {self.strides}')
                self.lemons.add(
                    Lemon(name=f'lemon{0}', multiplier=self.multiplier, in_channel=self.channels[0][-1],
                          channels=self.channels[1],
                          strides=self.strides[0], activation=self.activation, att1=self.att1, att2=self.att2)
                )
                for i in range(self.num_lemons-1):
                    self.lemons.add(
                        Lemon(name=f'lemon{i+1}', multiplier=self.multiplier, in_channel=self.channels[1][-1],
                                            channels=self.channels[1], strides=self.strides[0], activation=self.activation,
                                            att1=self.att1, att2=self.att2)
                    )
                # self.lemon2 = Lemon(name='lemon2', multiplier=self.multiplier, in_channel=self.channels[1][-1], channels=self.channels[1], strides=self.strides, activation=self.activation, att1=self.att1, att2=self.att2)
                # self.lemon3 = Lemon(name='lemon3', multiplier=self.multiplier, in_channel=self.channels[1][-1], channels=self.channels[1], strides=self.strides, activation=self.activation, att1=self.att1, att2=self.att2)
            self.output = Output(multiplier=self.multiplier, channels=self.channels[2], strides=self.strides[1], keypoints=self.keypoints, num_keypoints=self.num_keypoints, activation=self.activation)

    def hybrid_forward(self, F, x):
        #  128 X 128 --> 64 X 64
        x = self.input(x)
        print(f'\n\n\n')
        #  64 X 64 --> 32 X 32
        x = self.lemons(x)
        print(f'\n\n\n')
        # lemon1 = self.lemon1(input)
        # print(f'\n\n\n')
        # #  32 X 32 --> 16 X 16
        # lemon2= self.lemon2(lemon1)
        # print(f'\n\n\n')
        # #  16 X 16 --> 8 X 8
        # lemon3 = self.lemon3(lemon2)
        # print(f'\n\n\n')
        #  8 X 8 --> 4 X 4 --> (136, 2*num_keypints)
        output = self.output(x)
        return output





def generate_init_params(
        data_shape=(3, 128, 128),
        multiplier=1.0,
        channels=[[64, 128], [256, 256, 256], [128, 136]],
        strides=[[2, 1, 1], [2, 2, 1]],
        num_lemons=3,
        activation='sigmoid',
        att1=True,
        att2=True,
        keypoints=True,
        num_keyppoints=1,
        prefix='../checkpoints/FM/v3/ok/init',
        epoch=0
):
    net = IQ(
        multiplier=multiplier,
        channels=channels,
        strides=strides,
        num_lemons=num_lemons,
        activation=activation,
        att1=att1,
        att2=att2,
        keypoints=keypoints,
        num_keyppoints=num_keyppoints
    )
    net.initialize(force_reinit=True)
    # print(f'net: \n{net}')

    W, H = data_shape[2], data_shape[2]
    x = mx.random.uniform(shape=(2, 3, W, H))
    net.summary(x)

    y = net(x)
    print(f'[main] y: {len(y)}  {y[0].shape}  {y[1].shape if y[1] is not None else None}')

    net.hybridize()
    y = net.forward(x)
    if keypoints:
        print(f'[main-sym] y: {len(y)}  {y[0].shape}  {y[1].shape if y[1] is not None else None}')
    else:
        print(f'[main-sym] y: {y.shape}')

    net.export(prefix, epoch)

    sym, _, _ = mx.model.load_checkpoint(prefix=prefix, epoch=epoch)
    digraph = mx.viz.plot_network(sym, title=prefix+'-v3', save_format='pdf', shape={'data':(1, 3, W, H)})
    digraph.view()

    from model_zoo.utils import TransformGluon2Sym
    TransformGluon2Sym(prefix, epoch=0, data_shape=(3, W, H))




if __name__ == '__main__':


    att1 = True
    att2 = True
    keypoints = True
    num_keyppoints = 1
    MINI = False
    WIDTH = 128  # 128 / 64
    ext = f'{WIDTH}'
    multiplier = 1.0
    num_lemons = 1
    if MINI:
        channels = [[64, 128], [128, 128, 128], [128, 136]]
        ext = ext + '-mini-' + str(multiplier)
    else:
        channels = [[64, 128], [256, 256, 256], [128, 136]]
        ext = ext + '-normal-' + str(multiplier)
    if att1:
        strides = [[[2, 2, 1], [2, 2, 2], [2, 2, 2]], [2, 2, 1]] if WIDTH == 128 else [[[2, 1, 1], [2, 2, 1], [2, 2, 2]], [2, 2, 1]]
    else:
        strides = [[[2, 1, 1], [1, 1, 1], [2, 2, 2]], [2, 2, 1]] if WIDTH == 128 else [[[2, 1, 1], [1, 1, 1], [2, 2, 2]], [2, 2, 1]]

    print(f'strides: {strides}')
    data_shape = (3, WIDTH, WIDTH)
    activation = 'sigmoid'
    prefix = f'../checkpoints/FM/v3/ok/init-{ext}'
    epoch = 0

    generate_init_params(
        data_shape=data_shape,
        multiplier=multiplier,
        channels=channels,
        strides=strides,
        num_lemons=num_lemons,
        activation=activation,
        att1=att1,
        att2=att2,
        keypoints=keypoints,
        num_keyppoints=num_keyppoints,
        prefix=prefix,
        epoch=epoch
    )