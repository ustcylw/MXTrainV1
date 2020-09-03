#! /usr/bin/env python
# coding: utf-8
import numpy as np
import subprocess
import os, sys


# video_file = '/media/bill/cc18c1ef-ab04-47e4-ad1f-f431ab2785a3/datasets/300W/test/src/001/vid.avi'
# save_dir = '/media/bill/cc18c1ef-ab04-47e4-ad1f-f431ab2785a3/datasets/300W/test/crop'
# # cmd = 'ffmpeg -i {} -r {} -f pic-%03d.jpg'.format(video_file, save_dir)
# cmd = 'ffmpeg -i {} {}/pic-%03d.jpg'.format(video_file, save_dir)
# os.system(cmd)





def extract(root_dir, save_dir):
    videos = {}
    sub_dir_names = os.listdir(root_dir)
    for idx, sub_dir_name in enumerate(sub_dir_names):
        sub_dir = os.path.join(root_dir, sub_dir_name)
        annot_dir = os.path.join(sub_dir, 'annot')
        print('[{}/{}] dealing with {}'.format(idx, len(sub_dir_names), sub_dir_name))
        if not os.path.isdir(sub_dir):
            continue
        video_file = os.path.join(sub_dir, 'vid.avi')
        save_path = os.path.join(save_dir, sub_dir_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cmd = 'ffmpeg -i {} {}/%06d.jpg'.format(video_file, save_path)
        os.system(cmd)
        os.system('cp {}/*.pts {}'.format(annot_dir, save_path))



if __name__ == '__main__':

    root_dir = '/media/bill/cc18c1ef-ab04-47e4-ad1f-f431ab2785a3/datasets/300vw/src/300VW_Dataset_2015_12_14'
    save_dir = '/media/bill/cc18c1ef-ab04-47e4-ad1f-f431ab2785a3/datasets/300vw/images'

    extract(root_dir, save_dir)
