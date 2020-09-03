#! /usr/bin/env python
# coding: utf-8
import os, sys





def split_pts_file(pts_files, root_dir):
    for pts_file in pts_files:
        with open(pts_file, 'r') as lf:
            lines = None
            lines = lf.readlines()
        for line in lines:
            sub_path, points = line.strip().split(' ')
            print('sub_path: ', sub_path)
            print('points: ', points)
            points = points.replace(',', '\t')
            points = points + '\n'
            image_file = os.path.join(root_dir, sub_path)
            if not os.path.exists(image_file):
                print('image file is not exist!!!')
                continue
            point_file = image_file.replace('.png', '.pts')
            if os.path.exists(point_file):
                os.system('rm {}'.format(point_file))
            # with open(point_file, 'w+', encoding='utf-8') as df:
            with open(point_file, 'w') as df:
                df.write(points)
                df.flush()


if __name__ == '__main__':

    pts_files = [
        '/media/bill/cc18c1ef-ab04-47e4-ad1f-f431ab2785a3/datasets/crop_300w_128_/01_Indoor.pts',
        '/media/bill/cc18c1ef-ab04-47e4-ad1f-f431ab2785a3/datasets/crop_300w_128_/02_Outdoor.pts'
    ]
    root_dir = '/media/bill/cc18c1ef-ab04-47e4-ad1f-f431ab2785a3/datasets/crop_300w_128_'

    split_pts_file(pts_files, root_dir)