#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-03-07 11:53
# @Author  : zs

import re
import os
import glob

def list_file(root_pth, pattern_str=None):
    '''
    循环遍历目录,返回后缀str的list
    :param root_pth: 需要遍历的目录
    :param pattern_str: 后缀名
    :return: 匹配后缀名的list
    '''
    myfiles =[]

    for root, dirnames, filenames in os.walk(root_pth):
        for filename in filenames:
            if pattern_str is not None:
                if(filename.endswith(pattern_str)):
                    myfiles.append("%s/%s" % (root, filename))
            else:
                myfiles.append("%s/%s" % (root, filename))
    return myfiles

def search_file(data_dir, pattern=r'\.jpg$'):
    root_dir = os.path.abspath(data_dir)
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if re.search(pattern, f, re.I):
                abs_path = os.path.join(root, f)
                #print('new file %s' % absfn)
                yield abs_path


if __name__ == '__main__':
    pass
