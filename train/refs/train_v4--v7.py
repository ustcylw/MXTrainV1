#! /usr/bin/env python
# coding: utf-8
from __future__ import division
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("tf20_3DFR")+len("tf20_3DFR")] # 获取myProject，也就是项目的根路径
sys.path.insert(0, rootPath)
sys.path.insert(0, rootPath+'/tf_packages')
import tf_packages as TFPackages
import argparse
import numpy as np
import math
# import matplotlib.pyplot as plt
import tf_packages.dataset.tf_dataset_v7 as TFDatasetV7
#import tf_packages.models.hourglass as Hourglass
import tf_packages.models.resnet as Resnet
import tf_packages.loss.loss_v1 as LossV1
import tf_packages.log.logger as Logger
import tf_packages.utils.check_device as CheckDevice

# tf.config.experimental_run_functions_eagerly(True)


class TrainWithMirroredStrategy():
    
    def __init__(self, opts):
        self.opts = opts
        self.strategy = tf.distribute.MirroredStrategy()  # self.strategy = tf.distribute.MirroredStrategy(devices=['/gpu:0', '/gpu:1'])
        print ('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
        self.batch_size_per_replica = opts['batch_size']
        self.global_batch_size = self.batch_size_per_replica * self.strategy.num_replicas_in_sync
        self.start_epoch = opts['start_epoch']
        self.stop_epoch = opts['stop_epoch']
        self.train_root_dir = opts['train_root_dir']  # '/data2/datasets/300VW/test'
        self.train_image_list_file = opts['train_image_list_file']  # '/data2/datasets/300VW/test_facerecognition.txt'
        self.nrof_classes = opts['nrof_classes']
        
        print('='*80)
        print('[PARAMS]')
        print('  batch size per-replica: ', self.batch_size_per_replica)
        print('  global batch size: ', self.global_batch_size)
        print('  start epoch: ', self.start_epoch)
        print('  stop epoch: ', self.stop_epoch)
        print('  train root dir: ', self.train_root_dir)
        print('  train image list file: ', self.train_image_list_file)
        print('  number of class: ', self.nrof_classes)
        print('='*80)

    def Init(self):
        
        ## log
        self.logger = Logger.LogHandler(name='train_002', save_dir='/data2/personal/tf20')
        
        ## check device
        self.gpu_num, self.gpu_names = CheckDevice.check_available_gpus()
        self.logger.info('[======] {} GPUs detected {}'.format(self.gpu_num, self.gpu_names))
        
        ## dataset
        DataSet = TFDatasetV7.TFFolderDatasetV2(
            root_dir=self.train_root_dir,
            dataset_list=self.train_image_list_file,
            nrof_classes=self.nrof_classes,
            batch_size=self.batch_size_per_replica
        )
        dataset = DataSet.get_dataset()
        self.train_dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
        
        with self.strategy.scope():
            # 输入：[b, 32, 32, 3]
            self.model = Resnet.Resnet18(nrof_classes=self.nrof_classes)
            self.model.build(input_shape=(None, 32, 32, 3))
            #model.compile(
                #optimizer='adam', 
            #)
            #print('model: \n', model)
            self.model.summary()
            self.optimizer = optimizers.Adam(lr=1e-4)
            self.scc = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )
            self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train-accuracy')

    
    @tf.function
    def train_step(self, dist_inputs):
        def step_fn(self, inputs):
            features, labels = inputs
    
            with tf.GradientTape() as tape:
                logits = self.model(features)
                # print('[***] logits: ', logits.shape, logits.device)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                loss = tf.reduce_sum(cross_entropy) * (1.0 / self.global_batch_size)
                # cross_entropy = tf.losses.categorical_crossentropy(labels, logits, from_logits=True)
                # loss = tf.nn.compute_average_loss(cross_entropy, global_batch_size=self.global_batch_size)
                # print('[***] loss: ', loss.shape, loss)

            grads = tape.gradient(loss, self.model.trainable_variables)
            # self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
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
        #print('local result: ', len(ret), dir(ret[0]))
        #loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_example_losses, axis=None)
        # print('[===] loss: ', type(loss), dir(loss), loss.shape)
        # print('[===] loss: ', loss.shape, loss)
        return loss/self.global_batch_size
    
    
    def Train(self):

        import time
        global_step = 0
        start = time.time()
        step_interval = 1

        with self.strategy.scope():
            def compute_loss(labels, predictions):
                device_losses = self.scc(labels, predictions)
                device_loss = tf.nn.compute_average_loss(pre_example_loss=device_losses, global_batch_size=self.global_batch_size)
                return device_loss

            # with self.strategy.scope():
            def train_step(inputs):
                input_data, labels = inputs
                with tf.GradientTape() as tape:
                    logits = self.model(input_data)
                    loss = compute_loss(labels=labels, predictions=logits)
                gradients = tape.gradient(target=loss, sources=self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                self.train_accuracy.update_state(labels, logits)
                return loss

            # with self.strategy.scope()
            @tf.function
            def dist_train_step(dataset_inputs):
                replica_losses = self.strategy.experimental_run_v2(fn=train_step, args=(dataset_inputs,))
                print(f'replica losses: {replica_losses}')
                return self.strategy.reduce(reduce_op=tf.distribute.ReduceOp.SUM, value=replica_losses, axis=None)

            # with self.strategy.scope()
            for epoch in range(self.start_epoch, self.stop_epoch):
                epoch_loss = 0.0
                num_steps = 0
                for inputs in self.train_dist_dataset:
                    # print('inputs: ', type(inputs[0]), dir(inputs[0]), inputs[1])
                    step_loss = dist_train_step(inputs)
                    epoch_loss += step_loss
                    num_steps += 1
                    # print('loss: ', type(loss), type(loss.values), type(loss.values()[0]), loss)
                    if (num_steps + 1) % step_interval == 0:
                        end = time.time()
                        interval = end - start
                        start = end
                        print('[=============]', dir(inputs[0]))
                        print('[=============]', dir(inputs[0].values[0].shape))
                        self.logger.info('[{}/{}] loss: {}  {}  [{}--{}] [{}--{}]'.format(
                            self.stop_epoch-self.start_epoch,
                            epoch,
                            loss, 
                            # len(inputs), 
                            self.opts['batch_size'] * 10 / interval,
                            inputs[0].values[0].shape, 
                            inputs[0].values[1].shape, 
                            inputs[1].values[0].shape, 
                            inputs[1].values[1].shape
                        ))
                    global_step += 1


if __name__ == '__main__':
    
    opts = {
        'batch_size': 64,
        'stop_epoch': 30,
        'start_epoch': 0,
        # 'train_root_dir': '/data2/datasets/300VW/test',
        # 'train_image_list_file': '/data2/datasets/300VW/test_facerecognition.txt',
        'train_root_dir': '/data2/datasets/300VW/300VW_Dataset_2015_12_14',
        'train_image_list_file': '/data2/datasets/300VW/300VW_Dataset_2015_12_14_facerecognition.txt',
        'nrof_classes': 114
    }
    
    train_obj = TrainWithMirroredStrategy(opts)
    train_obj.Init()
    train_obj.Train()
