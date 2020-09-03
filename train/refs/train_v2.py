# coding: utf-8
import os, sys
import tensorflow as tf
import numpy as np
import cv2
from dataset.mnist import DataSource
from network.test_cnn import CNN


class CFG(object):
    batch_size = 128

cfg = CFG()


def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)


def trainning():
    # super params

    # model
    model = CNN()

    # loss
    loss_fn = tf.losses.SparseCategoricalCrossentropy()
    # difference ******
    # loss = tf.losses.categorical_crossentropy(y, predictions, from_logits=False)

    # optimizer

    # dataset
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Convert to float32.
    x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
    # Normalize images value from [0, 255] to [0, 1].
    x_train, x_test = x_train / 255., x_test / 255.
    # Use tf.data API to shuffle and batch data.
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.repeat().shuffle(5000).batch(cfg.batch_size).prefetch(1)

    # metric

    # lr-schedule, etc

    # if use_keras_fit:
    #     KerasFit.Train()
    # else:
    #     TrainLoop.Train()



if __name__ == '__main__':

    train()