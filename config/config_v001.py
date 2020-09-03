#! /usr/bin/env python
# coding: utf-8


class Config(object):
    def __init__(
        self,
        with_fit=True,
        gpus='0,1',
        pretrained_prefix='./model/eye-hole',
        pretrained_epoch=0,
        batch_size=24,
        log_interval=10,
        dtype='float32'
    ):
        self.log_interval = log_interval
        self.gpus = gpus
        self.batch_size = batch_size
        self.pretrained_prefix = pretrained_prefix
        self.pretrained_epoch = pretrained_epoch
        self.with_fit = with_fit
        self.dtype = dtype


class TrainConfig(Config):
    def __init__(
        self,
        with_fit=False,
        gpus='0,1',
        min_epoch=1,
        max_epoch=10,
        checkpoint_dir=r'../checkpoints',
        pretrained_prefix='../model_zoo/models/model-zoo',  # -sym',
        pretrained_epoch=0,
        batch_size=2,
        log_interval=1,
        weight_decay=1e-4,
        learning_rate=1e-3,
        data_shape=(3, 128, 128),
        label_shape=(2,),
        data_names=('data',),
        label_names=(('softmax_label'),),
        dtype='float32'
    ):
        super(TrainConfig, self).__init__(
            with_fit=with_fit,
            gpus=gpus,
            pretrained_prefix=pretrained_prefix,
            pretrained_epoch=pretrained_epoch,
            batch_size=batch_size,
            log_interval=log_interval,
            dtype=dtype
        )

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.data_names = data_names
        self.label_names = label_names
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.checkpoint_dir = checkpoint_dir


class EvalConfig(Config):
    def __init__(
        self,
        with_fit=False,
        gpus='0,1',
        pretrained_prefix='../model_zoo/models/model-zoo',  # -sym',
        pretrained_epoch=0,
        batch_size=2,
        log_interval=1,
        data_shape=(3, 128, 128),
        label_shape=(2,),
        data_names=('data',),
        label_names=(('softmax_label'),)
    ):
        super(EvalConfig, self).__init__(
            with_fit=with_fit,
            gpus=gpus,
            pretrained_prefix=pretrained_prefix,
            pretrained_epoch=pretrained_epoch,
            batch_size=batch_size,
            log_interval=log_interval
        )

        self.data_shape = data_shape
        self.label_shape = label_shape
        self.data_names = data_names
        self.label_names = label_names
