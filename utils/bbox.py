#! /usr/bin/env python
# coding: utf-8
import os, sys
import numpy as np
import cv2



def bbox_xyxy2xywh(bbox_xyxy):
    return np.array([
        [bbox_xyxy[0][0], bbox_xyxy[0][1]],
        [bbox_xyxy[1][0] - bbox_xyxy[0][0], bbox_xyxy[1][1] - bbox_xyxy[0][1]]
    ])


def bbox_xywh2xyxy(bbox_xywh):
    return np.array([
        [bbox_xywh[0][0], bbox_xywh[0][1]],
        [bbox_xywh[0][0] + bbox_xywh[1][0], bbox_xywh[0][1] + bbox_xywh[1][1]]
    ])


def square_bbox(bbox_xyxy, ori_shape):
    '''
    :param bbox_xyxy:
    :param ori_shape: [w, h]
    :return:
    '''
    bbox = np.array(bbox_xyxy)
    bbox = bbox.reshape((-1, 2))
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    center = np.array([(bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[1][1]) / 2])
    semi_edge = w / 2 if w > h else h / 2
    new_bbox = np.array([
        [center[0] - semi_edge, center[1] - semi_edge],
        [center[0] + semi_edge, center[1] + semi_edge]
    ])
    correct = True
    if (new_bbox < 0).any():
        correct = False
    if new_bbox[1][0] > ori_shape[0]:
        correct = False
    if new_bbox[1][1] > ori_shape[1]:
        correct = False

    if correct:
        return new_bbox

    semi_edge = w / 2 if w < h else h / 2
    new_bbox = np.array([
        [center[0] - semi_edge, center[1] - semi_edge],
        [center[0] + semi_edge, center[1] + semi_edge]
    ])
    new_bbox[new_bbox < 0] = 0
    if new_bbox[1][0] > ori_shape[0]:
        new_bbox[1][0] = ori_shape[0]
    if new_bbox[1][1] > ori_shape[1]:
        new_bbox[1][1] = ori_shape[1]

    return new_bbox


def random_scale_bbox(bbox_xyxy, scale=0.0):
    w, h = bbox_xyxy[1][0] - bbox_xyxy[0][0], bbox_xyxy[1][1] - bbox_xyxy[0][1]
    # print(f'[scale bbox] w: {w}  h: {h}')
    center = np.array([(bbox_xyxy[0][0] + bbox_xyxy[1][0]) / 2, (bbox_xyxy[0][1] + bbox_xyxy[1][1]) / 2])
    # print(f'[scale bbox] center: {center}')
    semi_w, semi_h = w / 2, h / 2
    # print(f'[scale bbox] semi-wh: {semi_w}  { semi_h}')
    semi_w, semi_h = semi_w * (1 + scale / 2), semi_h * (1 + scale / 2)
    # print(f'[scale bbox] semi-wh: {semi_w}  { semi_h}')
    new_bbox = np.array([
        [center[0] - semi_w, center[1] - semi_h],
        [center[0] + semi_w, center[1] + semi_h]
    ])
    return new_bbox
