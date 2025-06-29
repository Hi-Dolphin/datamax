# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'

import os
import sys
import time
import ctypes


dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'jiojio_cpp')

# load `get_pos_node_feature_c`，加载分词的特征抽取 C 优化函数
try:
    file_list = os.listdir(os.path.join(dir_path, 'build'))
    file_name = ''
    for _file_name in file_list:
        if 'libposFeatureExtractor' in _file_name and _file_name.endswith('.so'):
            file_name = _file_name
            break

    feature_extractor = ctypes.PyDLL(os.path.join(dir_path, 'build', file_name))
    get_pos_node_feature_c = feature_extractor.getPosNodeFeature
    get_pos_node_feature_c.argtypes = [
        ctypes.c_int, ctypes.py_object, ctypes.py_object,
        ctypes.py_object, ctypes.py_object]
    get_pos_node_feature_c.restype = ctypes.py_object

    # 统计若干次 C 处理特征的速度
    # 94.54%，96.53%，95.21%，94.63%，96.80%，96.93%，94.68%
    # C code 的处理速度大约为 py code 处理速度的 95.61%
    # 仍存在一定的改进空间，
except Exception:
    get_pos_node_feature_c = None


from .read_default_dict import ReadPOSDictionary
from .feature_extractor import POSFeatureExtractor
from .predict_text import POSPredictText
