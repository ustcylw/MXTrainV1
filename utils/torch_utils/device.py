#! /usr/bin/env python
# coding: utf-8
import os, sys
import torch
import numpy as np



def get_device(gpu_id=-1):
    '''
    # gpu_id: 
    #   torch.device
    #   int:
    #     -1: use cpu
    #     >=0: use gpu
    #   str: 
    #     '0': use gpu-0
    #     '0,1,2': use gpu-0, gpu-1, gpu-2
    # 
    '''
    if isinstance(gpu_id, torch.device):
        return [gpu_id]
    if isinstance(gpu_id, int):
        if gpu_id < 0:
            device = [torch.device('cpu', 0)]
        else:
            device = [torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu:0')]
    if isinstance(gpu_id, str):
        device = [torch.device(f'cuda:{int(i)}') for i in gpu_id.split(',')]
    return device


def to(data, device, dtype=torch.float32):
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)
    return data.to(device).astype(dtype)
