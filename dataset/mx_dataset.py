#! /usr/bin/env python
# coding: utf-8
import os, sys
import mxnet as mx
import numpy as np


a = np.array(np.linspace(0, 5*6*3-1, 5*6*3).reshape((5, 6, 3)))
labels_1 = [1, 2, 3, 4, 5]
print(f'a: {a}')

def pack_array(arr):
    header = mx.recordio.IRHeader(0, i, i+1, 0)
    s = arr.tobytes(order='c')
    packed_s = mx.recordio.pack(header, s)
    return packed_s

def unpack_array(packed_s):
    header, s = mx.recordio.unpack(packed_s)
    arr = np.fromstring(s, a.dtype)  # np.uint8)
    return arr


if False:
    record = mx.recordio.MXRecordIO('test.rec', 'w')
    for i in range(5):
        packed_s = pack_array(a+i)
        print(f'packed_s_{i}: {type(packed_s)}')
        record.write(packed_s)

    record = mx.recordio.MXRecordIO('test.rec', 'r')
    for i in range(5):
        packed_s = record.read()
        arr = unpack_array(packed_s)
        print(f'arr_{i}: {arr}')
else:
    rec_file = 'test_idx.rec'
    idx = 0
    idx_string = ''
    idx_file = 'test_idx.idx'
    # # record = mx.recordio.MXRecordIO(rec_file, 'w')
    # record = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'w')
    # for i in range(5):
    #     packed_s = pack_array(a+i)
    #     print(f'packed_s_{i}: {type(packed_s)}')
    #     # record.write(packed_s)
    #     record.write_idx(i, packed_s)
    #     idx_string += str(i) + '\t' + str(idx) + '\n'
    #     idx += len(packed_s)
    # with open(idx_file, 'w') as df:
    #     df.write(idx_string)
    #     df.flush()

    record = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')
    idx_lines = None
    idx_list = []
    with open(idx_file, 'r') as lf:
        idx_lines = lf.readlines()
    for line in idx_lines:
        line = line.strip()
        idx, cursor = line.split('\t')
        idx_list.append(cursor)
    for i in range(5):
        cursor = 5 - 1 - i  # int(idx_list[5-1-i])
        print(f'cursor: {cursor}')
        packed_s = record.read_idx(cursor)
        arr = unpack_array(packed_s)
        print(f'arr_{i}: {arr}')
