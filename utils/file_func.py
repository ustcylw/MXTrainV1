#! /usr/bin/env python
# coding: utf-8
import os, sys
import inspect


def get_function_name():
    '''get runtime function name'''
    return inspect.stack()[1][3]


def get_class_name(obj):
    '''get current class name'''
    return obj.__class__.__name__


if __name__ == "__main__":
    def yoyo():
        print("function name：%s" % get_function_name())


    class Yoyo():
        def yoyoketang(self):
            print("class_name.method_name： %s.%s" % (self.__class__.__name__, get_function_name()))


    yoyo()
    Yoyo().yoyoketang()

    import os
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # print(f'os.path.abspath(__file__): {os.path.abspath(__file__)}')
    # print(f'os.path.dirname(os.path.abspath(__file__)): {os.path.dirname(os.path.abspath(__file__))}')
    # print(f'base path: {base_path}')
    #
    # print(f'packages root dir: {get_package_dir()}')
    # print(f'project root dir: {get_project_dir()}')
