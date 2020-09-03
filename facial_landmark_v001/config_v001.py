#! /usr/bin/env python
# coding: utf-8
from config.config_v001 import TrainConfig, EvalConfig


class FMTrainConfig(TrainConfig):
    def __init__(
        self,
        with_fit='fit',
        gpus='0',
        min_epoch=200,
        max_epoch=250,
        checkpoint_dir=r'../checkpoints/FM/001/epochs/init-128-128',
        pretrained_prefix=r'../checkpoints/FM/001/epochs/init-128-128',  # -sym',
        pretrained_epoch=3,
        batch_size=64,
        log_interval=20,
        image_dir=r'/data2/datasets/300VW/300VW_Dataset_2015_12_14',
        eval_image_dir=r'/data2/datasets/300w-format',
        train_rec_dir=r'/data1/dataset/300/rec/002',
        train_rec_prefix=r'300vw_128X128',
        eval_rec_dir=r'/data1/dataset/300/rec/002',
        eval_rec_prefix=r'300w_128X128',
        weight_decay=1e-4,
        learning_rate=1e-3,
        data_shape=((3, 128, 128),),
        label_shape=((136,), (2,)),
        data_names=('data',),
        label_names=(('kpts68-regression', 'kptsn-regression')),
        loss_weights=[1.0, 1.0, 1.0],
        show=False,
        save=False,
        save_prefix='../results/',
        dtype='float32'
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
            data_shape=data_shape,
            label_shape=label_shape,
            data_names=data_names,
            label_names=label_names,
            dtype=dtype
        )

        self.image_dir = image_dir
        self.eval_image_dir = eval_image_dir
        self.loss_weights = loss_weights
        self.show = show
        self.save = save
        self.save_prefix = save_prefix
        self.train_rec_dir = train_rec_dir
        self.train_rec_prefix = train_rec_prefix
        self.eval_rec_dir = eval_rec_dir
        self.eval_rec_prefix = eval_rec_prefix


class FMEvalConfig(EvalConfig):
    def __init__(
        self,
        with_fit=False,
        gpus='0,1',
        pretrained_prefix=r'../checkpoints/FM/001/ok/init-128-128',  # -sym',
        pretrained_epoch=0,
        batch_size=2,
        log_interval=1,
        image_dir=r'/data2/datasets/300w-format',
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
        self.loss_weight = losss_weights



