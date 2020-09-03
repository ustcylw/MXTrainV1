# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class AccMetric(object):

    def __init__(self, name='acc-metric'):
        self.name = name
        self.key = 0.0
        self.val = 0.0
        self.avg = 0.0
        self.avg_cur = 0.0
        self.key_cur = 0.0
        self.val_cur = 0.0

    def update(self, key, val):
        '''
        :param key: key is labels, sparse label
        :param val: val is predictions
        :return:
        '''
        labels = key  # |B, |
        predictions = val  # |B, Features|
        preds = np.argmax(predictions, 1)
        correct_count = (preds == labels).sum()
        batch_size = labels.shape[0]
        self.key += batch_size
        self.val += correct_count
        self.key_cur = batch_size
        self.val_cur = correct_count
        self.avg = self.val / (1.0 if ((self.key > -1e-12) and (self.key < 1e-12)) else self.key)
        self.avg_cur = correct_count / (1.0 if ((batch_size > -1e-12) and (batch_size < 1e-12)) else batch_size)

    def get_avg(self):
        return self.avg, self.avg_cur

    def get_key_val(self):
        return self.key, self.val, self.key_cur, self.val_cur


if __name__ == '__main__':
    acc_metric = AccMetric()
    predictions = np.array(
        [
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.9],
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0, 0.0]
        ]
    )
    labels = np.array([0, 1, 2, 3, 4, 0, 0])
    acc_metric.update(labels, predictions)
    acc = acc_metric.get_avg()
    print(f'acc: {acc[0]}/{acc[1]}')
    acc_metric.update(labels[0:-1], predictions[0:-1, :])
    acc = acc_metric.get_avg()
    print(f'{acc_metric.name}: {acc[0]}/{acc[1]}')