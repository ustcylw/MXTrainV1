#! /usr/bin/env python
# coding: utf-8
from config.config_v3 import TrainConfig, EvalConfig


class FMTrainConfig(TrainConfig):
    def __init__(
        self,
        with_fit='fit',
        gpus='0,1',
        min_epoch=1,
        max_epoch=10,
        checkpoint_dir=r'../checkpoints',
        pretrained_prefix='../model_zoo/models/model-zoo',  # -sym',
        pretrained_epoch=0,
        transform_prefix=r'../model_zoo/models/model-zoo',
        transform_epoch=0,
        batch_size=2,
        log_interval=1,
        num_class=2,
        image_dir=r'/home/intellif/Desktop/北京活体检测/20200806/test',
        weight_decay=1e-4,
        learning_rate=1e-3,
        data_shape=(3, 128, 128),
        loss_weights=[1.0, 1.0]
    ):
        super(FMTrainConfig, self).__init__(
            with_fit=with_fit,
            gpus=gpus,
            min_epoch=min_epoch,
            max_epoch=max_epoch,
            checkpoint_dir=checkpoint_dir,
            pretrained_prefix=pretrained_prefix,
            pretrained_epoch=pretrained_epoch,
            batch_size=batch_size,
            log_interval=log_interval,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            data_shape=data_shape
        )

        self.num_class = num_class
        self.image_dir = image_dir
        self.transform_prefix = transform_prefix
        self.transform_epoch = transform_epoch
        self.loss_weights = loss_weights


class FMEvalConfig(EvalConfig):
    def __init__(
        self,
        with_fit=False,
        gpus='0,1',
        pretrained_prefix='../model_zoo/models/model-zoo',  # -sym',
        pretrained_epoch=0,
        transform_prefix=r'../model_zoo/models/model-zoo',
        transform_epoch=0,
        batch_size=2,
        log_interval=1,
        image_dir=r'/home/intellif/Desktop/北京活体检测/20200806/test',
        data_shape=(3, 128, 128),
        losss_weights=[1.0]
    ):
        super(FMEvalConfig, self).__init__(
            with_fit=with_fit,
            gpus=gpus,
            pretrained_prefix=pretrained_prefix,
            pretrained_epoch=pretrained_epoch,
            batch_size=batch_size,
            log_interval=log_interval,
            data_shape=data_shape
        )

        self.image_dir = image_dir
        self.transform_prefix = transform_prefix
        self.transform_epoch = transform_epoch
        self.loss_weight = losss_weights


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
        data_shape=(3, 128, 128)
    ):
        super(TrainConfig, self).__init__(
            with_fit=with_fit,
            gpus=gpus,
            pretrained_prefix=pretrained_prefix,
            pretrained_epoch=pretrained_epoch,
            batch_size=batch_size,
            log_interval=log_interval
        )

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.data_shape = data_shape
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
        data_shape=(3, 128, 128)
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
