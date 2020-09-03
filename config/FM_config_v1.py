#! /usr/bin/env python
# coding: utf-8
from config.config_v1 import Config


class Config1(Config):
    def __init__(
        self,
        with_fit=False,
        gpus='0,1',
        min_epoch=1,
        max_epoch=10,
        checkpoint_dir=r'../checkpoints/FM/v1',
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
        data_shape=(3, 128, 128)
    ):
        super(Config1, self).__init__(
            with_fit=with_fit,
            gpus=gpus,
            min_epoch=min_epoch,
            max_epoch=max_epoch,
            checkpoint_dir=checkpoint_dir,
            pretrained_prefix=pretrained_prefix,
            pretrained_epoch=pretrained_epoch,
            batch_size=batch_size,
            log_interval=log_interval
        )

        self.num_class = num_class
        self.image_dir = image_dir
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.data_shape = data_shape
        self.transform_prefix = transform_prefix
        self.transform_epoch = transform_epoch
