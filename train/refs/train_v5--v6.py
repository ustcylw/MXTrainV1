#! /usr/bin/env python
# coding: utf-8
from __future__ import division
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import argparse
import numpy as np
import math
# import dataset.tfrecord_v1 as TFRecordDataset
from dataset.tfrecord_v1 import TFRecordLandmark
import network.resnet as Resnet
from network.resnet import Resnet18
import log.logger as Logger
import utils.check_device as CheckDevice
from config.config_v1 import Config
from log.logger_v1 import LogHandler
from device_control.device_info import set_gpu_limit, list_physical_devices


# tf.config.experimental_run_functions_eagerly(True)

class TrainWithMirroredStrategy():
    
    def __init__(self, opts):
        self.opts = opts
        print('='*80)
        print('[PARAMS]')
        print(self.opts.to_string())
        print('='*80)

        ## device list
        devices = ['/GPU:0', '/GPU:1']
        physical_devices = list_physical_devices()
        print(f'physical_devices: {physical_devices}')

        ## set gpu limit
        set_gpu_limit(physical_devices, memory_limit=1024*5)

        # self.strategy = tf.distribute.MirroredStrategy(devices=['/device:XLA_GPU:0', '/device:XLA_GPU:1'])
        # self.strategy = tf.distribute.MirroredStrategy(devices=['/device:GPU:0', '/device:GPU:1'])
        # self.strategy = tf.distribute.MirroredStrategy(devices=devices)
        self.strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
        self.batch_size_per_replica = opts.batch_size
        self.global_batch_size = self.batch_size_per_replica * self.strategy.num_replicas_in_sync
        self.start_epoch = opts.min_epoch
        self.stop_epoch = opts.max_epoch
        self.tfrecord_file = opts.tfrecord_file
        self.num_classes = opts.num_classes

    def Init(self):
        
        ## log
        self.logger = LogHandler(name='train_002', save_dir='/data2/personal/tf20')

        ## check device
        self.gpu_num, self.gpu_names = CheckDevice.check_available_gpus()
        # self.gpu_num /= 2
        self.logger.info('[======] {} GPUs detected {}'.format(self.gpu_num, self.gpu_names))
        
        ## dataset
        dataset = TFRecordLandmark.tfrecord_loader(
            tfrecord_file=self.opts.tfrecord_file,
            shuffle=self.opts.shuffle,
            shuffle_size=self.opts.shuffle_size,
            batch_size=self.global_batch_size  # self.opts.batch_size
        )
        # TFRecordLandmark.display_tfrecord(dataset)

        with self.strategy.scope():
            # wrapper dist-dataset
            self.train_dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
            # 输入：[b, 128, 128, 3]
            self.model = Resnet18(num_classes=self.opts.num_classes)
            self.model.build(input_shape=(None, 128, 128, 3))
            #model.compile(
                #optimizer='adam', 
            #)
            self.optimizer = optimizers.Adam(lr=self.opts.learning_rate)

            ## checkpoint
            self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

            ## metrics
            self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        # print('model: \n', model)
        self.model.summary()

    
    @tf.function
    def train_step(self, dist_inputs):
        with self.strategy.scope():
            # CE = tf.keras.losses.SparseCategoricalCrossentropy(
            #     from_logits=True,
            #     reduction=tf.keras.losses.Reduction.NONE
            # )
            # CE = tf.keras.losses.categorical_crossentropy()
            CE = tf.keras.losses.CategoricalCrossentropy(
                from_logits=False,
                label_smoothing=0,
                reduction=tf.keras.losses.Reduction.NONE,
                name='categorical_crossentropy'
            )
            def step_fn(self, inputs):
                features, labels = inputs
                with tf.GradientTape() as tape:
                    logits = self.model(features)
                    # print('[***] logits: ', logits.shape, logits.device)
                    # print('[***] labels: ', labels.shape, labels.device)
                    # cross_entropy = CE(logits, labels)  # tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                    cross_entropy = tf.keras.losses.MSE(labels, logits)  # CE(labels, logits)  # tf.keras.losses.categorical_crossentropy(labels, logits)
                    # loss = tf.reduce_sum(cross_entropy) * (1.0 / self.global_batch_size)
                    loss = tf.nn.compute_average_loss(cross_entropy, global_batch_size=self.global_batch_size)
                    # print('[***] loss: ', loss.shape, loss)

                grads = tape.gradient(loss, self.model.trainable_variables)
                # self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                self.train_accuracy.update_state(labels, logits)
                return loss
    
        per_example_losses = self.strategy.experimental_run_v2(step_fn, args=(self, dist_inputs))
        #print('per example losses: ',
            #len(per_example_losses.values),
            #per_example_losses.values[0].shape,
            #per_example_losses.values[1].shape,
        #)
        # nrof_samples = 0.0
        # for i in range(len(per_example_losses.values)):
        #     nrof_samples += per_example_losses.values[i].shape[0]
        #ret = self.strategy.experimental_local_results(per_example_losses)
        #loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=None)
        # return loss/self.global_batch_size
        # self.train_loss.update_state(loss)
        return loss
    
    def Train(self):

        import time
        step = 0
        start = time.time()
        step_interval = 1
        with self.strategy.scope():
            for epoch in range(self.start_epoch, self.stop_epoch):
                for inputs in self.train_dist_dataset:
                    # print('inputs: ', inputs, type(inputs[0]), dir(inputs[0]))
                    loss = self.train_step(inputs)
                    # print('loss: ', type(loss), type(loss.values), type(loss.values()[0]), loss)
                    self.train_loss.update_state(loss.numpy())
                    if step % step_interval == 0:
                        end = time.time()
                        interval = end - start
                        start = end
                        # print('[=============]', dir(inputs[0]))
                        # print('[=============]', dir(inputs[0].values[0].shape))
                        # self.logger.info('[{}/{}] loss: {}  {}  [{}--{}] [{}--{}]'.format(
                        #     self.stop_epoch-self.start_epoch,
                        #     epoch,
                        #     loss,
                        #     # len(inputs),
                        #     self.opts.batch_size* 10 / interval,
                        #     inputs[0].values[0].shape,
                        #     inputs[0].values[1].shape,
                        #     inputs[1].values[0].shape,
                        #     inputs[1].values[1].shape
                        #     # inputs[0].numpy().shape,
                        #     # inputs[0].numpy().shape,
                        #     # inputs[1].numpy().shape,
                        #     # inputs[1].numpy().shape
                        # ))
                        self.logger.info('[{}/{}] loss: {}  {}'.format(
                            self.stop_epoch - self.start_epoch,
                            epoch,
                            loss,
                            self.train_loss.result()
                        ))
                    step += 1


if __name__ == '__main__':
    
    opts = Config()
    
    train_obj = TrainWithMirroredStrategy(opts)
    train_obj.Init()
    train_obj.Train()
