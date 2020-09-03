#! /usr/bin/env python
# coding: utf-8
import os, sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time
import threading
from concurrent.futures import ThreadPoolExecutor


def pusher(q):
    for i in range(5):
        q.put(i)
        print('running thread-{}:{}'.format(threading.get_ident(), i))
        time.sleep(1)

def poper(q):
    for i in range(5):
        msg = q.get()
        print('running thread-{}:{}  -->  {}'.format(threading.get_ident(), i, msg))
        time.sleep(1)


q = Queue()
pool = ThreadPoolExecutor(2)
pool.submit(pusher, q)
pool.submit(poper, q)
q.join()