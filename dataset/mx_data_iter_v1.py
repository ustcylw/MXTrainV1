#! /usr/bin/env python
# coding: utf-8
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import mxnet as mx
import numpy as np
import cv2
import pickle
from mx_rec import MXRec

class RecDataIterV1(object):
    def __init__(
            self,
            rec_dir,
            rec_prefix,
            batch_size=0,
            shuffle=True,
            aug_list=None,
            devices=None
    ):
        self.rec_dir = rec_dir
        self.rec_prefix = rec_prefix
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cursor = 0
        self.aug_list = aug_list
        self.devices = devices if devices is not None else mx.cpu()
        self.idx_list = []
        self.rec_handler = MXRec(rec_dir=self.rec_dir, prefix=self.rec_prefix)
        self.parse_idx_file()
        self.max_index = len(self.idx_list)

    def __iter__(self):
        return self

    def reset(self):
        self.cursor = 0

    def next(self):
        if self.iter_next():
            data = self.getdata()
            return mx.io.DataBatch(data=[data[0]], label=[data[1]], pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        if self.cursor + self.batch_size > self.max_index:
            return False
        return True

    def getdata(self):
        data, target = None, None
        if self.iter_next():
            batch_items = self.idx_list[self.cursor:(self.cursor+self.batch_size)]
            batch_data = []
            for batch_item in batch_items:
                batch_data.append(self.rec_handler.read_rec(batch_item))
            data_list, target_list = [], []
            for i in range(len(batch_data)):
                image = batch_data[i]['image']
                profile = batch_data[i]['profile']
                points = batch_data[i]['points']
                # print('image: {}'.format(image.shape))
                # print('profile: {}  {}'.format(profile.shape, profile.dtype))
                # print('points: {}'.format(points.shape))
                # image = image[np.newaxis, :, :, :]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.transpose(image, (2, 0, 1))
                # profile = profile[:, :, 2]
                profile = profile[np.newaxis, :, :]
                image = np.concatenate((image, profile), 0)
                # print('image: {}'.format(image.shape))
                # print('profile: {}'.format(profile.shape))
                if self.aug_list:
                    for aug in self.aug_list:
                        image = aug.process(image)
                data_list.append(image)
                target_list.append(points)
            data_array = np.array(data_list)
            target_array = np.array(target_list)
            data = mx.nd.array(data_array, ctx=self.devices)
            target = mx.nd.array(target_array, ctx=self.devices)
        else:
            raise StopIteration
        # return mx.io.DataBatch(data=[data], label=[target])
        self.cursor += self.batch_size
        return data, target

    def getlabel(self):
        pass

    def getindex(self):
        return None

    def getpad(self):
        pass

    def parse_idx_file(self):
        if len(self.idx_list) > 0:
            self.idx_list.clear()
        idx_file = os.path.join(self.rec_dir, self.rec_prefix+'.idx')
        with open(idx_file, 'r') as lf:
            lines = lf.readlines()
        for line in lines:
            line = line.strip()
            idx, num_bytes = line.split('\t')
            idx, num_bytes = int(idx), int(num_bytes)
            self.idx_list.append(idx)



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    rec_dir = './'
    rec_prefix = '300vw_3_2'
    batch_size = 2
    shuffle = True
    loader = RecDataIterV1(
        rec_dir=rec_dir,
        rec_prefix=rec_prefix,
        batch_size=batch_size,
        shuffle=shuffle
    )

    for index, data in enumerate(iter(loader)):
        images = data.data[0]
        labels = data.label[0]
        # print(f'images: {images.shape}')
        # print(f'labels: {labels.shape}')
        shape = images.shape
        for i in range(shape[0]):
            image = images[i, :, :, :].asnumpy()
            profile = image[3, :, :]
            image = image[0:3, :, :]
            label = labels[i, :].reshape((68, 2)).asnumpy()
            image = np.transpose(image, (1, 2, 0))
            image = image.astype(np.uint8)
            image_ = image
            profile = cv2.cvtColor(profile, cv2.COLOR_GRAY2BGR)
            profile = profile.astype(np.uint8)
            print('image: {}  {}'.format(image.shape, image.dtype))
            print('profile: {}  {}'.format(profile.shape, profile.dtype))
            image = cv2.addWeighted(image, 0.5, profile, 0.5, 0)
            for j in range(68):
                point = label[j, :]
                image = cv2.circle(
                    img=image,
                    center=(int(round(point[0])), int(round(point[1]))),
                    radius=1,
                    color=(255, 0, 0),
                    thickness=-1
                )
            print('image: {}  {}  {}'.format(image.shape, image.dtype, type(image)))
            # cv2.imshow('image', image)
            # if cv2.waitKey(1000) == ord('q'):
            #     sys.exit(0)
            # print('[{}/{}/{}]  image: {}  {}'.format(index, i, j, type(image.get()), image.get().shape))
            print('[{}/{}/{}]  image: {}  {}'.format(index, i, j, type(image), image.shape))
            # plt.imshow(image.get())  ## cv2.UMat --> np.array
            plt.imshow(image)  ## cv2.UMat --> np.array
            plt.show()
