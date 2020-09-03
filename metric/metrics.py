# -*- coding: utf-8 -*-


class BaseMetric(object):

    def __init__(self, name):
        self.name = name
        self.key = 0.0
        self.val = 0.0
        self.avg = 0.0
        self.avg_cur = 0.0
        self.key_cur = 0.0
        self.val_cur = 0.0

    def update(self, key, val):
        self.key += key
        self.val += val
        self.key_cur = key
        self.val_cur = val
        self.avg = self.val / (self.key if ((self.key > -1e-12) and (self.key < 1e12)) else 1.0)
        self.avg_cur = val / (key if ((key > -1e-12) and (key < 1e12)) else 1.0)

    def get_avg(self):
        # print('[===]  {} --> {}'.format(self.key, self.val))
        return self.avg, self.avg_cur

    def get_key_val(self):
        return self.key, self.val, self.key_cur, self.val_cur