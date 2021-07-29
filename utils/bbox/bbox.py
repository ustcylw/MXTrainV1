#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import torch


class BBoxes(object):
    '''
    # all data type is numpy.ndarray
    # field:
    #   {
    #     'field-name': np.ndarray
    #   }
    '''
    def __init__(self, bboxes, mode="xyxy", device=torch.device('cpu:0')):
        if bboxes.ndim != 2:
            raise ValueError(
                f"bbox should have 2 dimensions, got {bboxes.ndim}"
            )
        if bboxes.shape[-1] != 4:
            raise ValueError(
                f"last dimension of bbox should have a "
                "size of 4, got {bboxes.shape[-1]}"
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bboxes = bboxes
        self.shape = bboxes.shape
        self.mode = mode
        self.extra_fields = {}
        self.device = device

    def add_field(self, field, field_data):
        # if self.bboxes.shape[0] != field_data.shape[0]:
        #     raise ValueError(f'field-data shape {field_data.shape} is not equal to bboxes-shape {self.bboxes.shape}')
        self.extra_fields[field] = field_data

    def update_field(self, field, field_data):
        # if self.bboxes.shape[0] != field_data.shape[0]:
        #     raise ValueError(f'field-data shape {field_data.shape} is not equal to bboxes-shape {self.bboxes.shape}')
        self.extra_fields.update({field:field_data})

    def get_field(self, field):
        if not self.has_field(field):
            return np.array([])
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def __getitem__(self, item):
        if item >= self.bboxes.shape[0]:
            raise StopIteration
        bbox = BBoxes(self.bboxes[item:(item+1), :], self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, [v[item:(item+1), :]])
        return bbox

    def __len__(self):
        return self.bboxes.shape[0]

    def area(self):
        bboxes = self.bboxes
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (bboxes[:, 2] - bboxes[:, 0] + TO_REMOVE) * (bboxes[:, 3] - bboxes[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = bboxes[:, 2] * bboxes[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def centers(self):
        bboxes = self.bboxes
        center_points = np.zeros(shape=(bboxes.shape[0], 2))
        if self.mode == 'xyxy':
            center_points[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
            center_points[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
        else:
            center_points[:, 0] = bboxes[:, 0] + bboxes[:, 2] / 2
            center_points[:, 1] = bboxes[:, 1] + bboxes[:, 3] / 2
        return center_points

    def scale(self, scale):
        if scale == 1.0:
            return self
        
        if self.mode == 'xywh':
            delta_w = self.bboxes[:, 2] * (scale - 1.0) / 2
            delta_h = self.bboxes[:, 3] * (scale - 1.0) / 2
            self.bboxes[:, 2] *= scale
            self.bboxes[:, 3] *= scale
            self.bboxes[:, 0] -= delta_w
            self.bboxes[:, 1] -= delta_h
        else:
            # print(f'\n==>  bboxes: \n{self.bboxes}\n')
            delta_w = (self.bboxes[:, 2] - self.bboxes[:, 0]) * (scale - 1.0) / 2
            delta_h = (self.bboxes[:, 3] - self.bboxes[:, 1]) * (scale - 1.0) / 2
            # print(f'==>  delta_w: \n{delta_w}')
            # print(f'==>  delta_h: \n{delta_h}\n')
            self.bboxes[:, 0] -= delta_w
            self.bboxes[:, 1] -= delta_h
            self.bboxes[:, 2] += delta_w
            self.bboxes[:, 3] += delta_h
            # print(f'==>  bboxes: {self.bboxes}\n')
        return self
            
    def copy_with_fields(self):
        bbox = BBoxes(self.bboxes, self.mode, self.device)
        
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v)
            
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "mode={}".format(self.mode)
        s += "bboxes={})".format(self.mode)
        return s

    def to_xywh(self):
        if self.mode == 'xywh':
            return self
        
        self.bboxes[:, 2] = self.bboxes[:, 2] - self.bboxes[:, 0]
        self.bboxes[:, 3] = self.bboxes[:, 3] - self.bboxes[:, 1]
        self.mode = 'xywh'

        return self
    
    def to_xyxy(self):
        if self.mode == 'xyxy':
            return self
        
        self.bboxes[:, 2] = self.bboxes[:, 2] + self.bboxes[:, 0]
        self.bboxes[:, 3] = self.bboxes[:, 3] + self.bboxes[:, 1]
        self.mode = 'xyxy'

        return self

    def format(self, w, h):
        self.bboxes[self.bboxes<0] = 0
        
        if self.mode == 'xyxy':
            self.bboxes[self.bboxes[:, 2]>w, 2] = w
            self.bboxes[self.bboxes[:, 3]>h, 3] = h
        elif self.mode == 'xywh':
            self.bboxes[self.bboxes[:, 2]>w, 2] = w
            self.bboxes[self.bboxes[:, 3]>h, 3] = h
    
    def crop(self, feature):
        '''
        #  同一个feature上的bboxes
        '''
        rois = []
        for bbox in self.bboxes:
            bbox = bbox.astype(np.int32)
            if self.mode == 'xyxy':
                rois.append(feature[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]])
            elif self.mode == 'xywh':
                rois.append(feature[:, :, bbox[0]:(bbox[0]+bbox[2]), bbox[1]:(bbox[1]+bbox[3])])
            else:
                raise ValueError("should not reach here!!!  mode: {self.mode}")
        return rois
    

if __name__ == "__main__":
    
    bboxes = np.array([[100, 75, 245, 250], [288, 92, 431, 224], [541, 43, 682, 213], [701, 66, 868, 276]])
    print(f'bboxes: {type(bboxes)}  {bboxes.shape}')
    
    bb = BBoxes(bboxes, mode='xyxy')
    lm = np.array([
            [1,2,3,4,5,6,7,8,9,0],
            [2,3,4,5,6,7,8,9,0,1],
            [3,4,5,6,7,8,9,0,1,2],
            [4,5,6,7,8,9,0,1,2,3]
    ])
    bb.update_field(r'face_landmarks', lm)
    print(f'bb: {bb}')
    
    print(f'-'*80)
    print(f'-'*80)
    for idx, bbox in enumerate(bb):
        print(f'[{idx}]  bbox: {bbox.bboxes}')
        print(f'[{idx}]  face_landmarks: {bbox.get_field("face_landmarks")}')
        print(f'[{idx}]  centers: {bbox.centers()}')
        print(f'[{idx}]  area: {bbox.area()}')
        bbox = bbox.to_xyxy()
        print(f'[{idx}]  bbox-xyxy: {bbox.bboxes}')
        bbox = bbox.to_xywh()
        print(f'[{idx}]  bbox-xywh: {bbox.bboxes}')
        print(f'='*80)
    print(f'-'*80)
    print(f'-'*80)
    
    new_bboxes = bb.copy_with_fields()

    print(f'*'*80)
    print(f'-'*80)
    print(f'-'*80)
    for idx, bbox in enumerate(bb):
        print(f'[{idx}]  bbox: {bbox.bboxes}')
        print(f'[{idx}]  face_landmarks: {bbox.get_field("face_landmarks")}')
        print(f'[{idx}]  centers: {bbox.centers()}')
        print(f'[{idx}]  area: {bbox.area()}')
        bbox = bbox.to_xyxy()
        print(f'[{idx}]  bbox-xyxy: {bbox.bboxes}')
        bbox = bbox.to_xywh()
        print(f'[{idx}]  bbox-xywh: {bbox.bboxes}')
        print(f'='*80)
    print(f'-'*80)
    print(f'-'*80)
