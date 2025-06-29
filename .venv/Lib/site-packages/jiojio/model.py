# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'

import os
import pdb
import sys
import json
import numpy as np

from jiojio import logging


class Model(object):
    def __init__(self, config, n_feature, n_tag, task=None):
        # task 可取 cws 或 pos，其用意在于针对不同的模型有不同的压缩方法
        # 当 task 取 None 时，不采用任何压缩方法，直接存储为文件
        self.task = task
        self.config = config
        self.n_tag = n_tag
        self.n_feature = n_feature

        if self.config.random_init:
            # 均值为 0 的均匀分布
            # self.w = np.random.random(
            #     size=(self.n_transition_feature,)) * 2 - 1

            # 正太分布更加符合真实情况
            self.node_weight = np.random.normal(
                loc=0.0, scale=0.4, size=(self.n_feature, self.n_tag))
            self.edge_weight = np.random.normal(
                loc=0.0, scale=0.4, size=(self.n_tag, self.n_tag))

            self.node_weight = self.node_weight.astype(np.float32)
            self.edge_weight = self.edge_weight.astype(np.float32)
            self.bi_ratio = np.array(np.random.random(), dtype=np.float32)  # 介于 0 ~ 1 的 float
            # self.bi_ratio = 0.5
        else:
            self.node_weight = np.zeros((self.n_feature, self.n_tag), dtype=np.float32)
            self.edge_weight = np.zeros((self.n_tag, self.n_tag), dtype=np.float32)
            self.bi_ratio = np.array(0.0001, dtype=np.float32)

    @classmethod
    def load(cls, model_dir, task=None, dtype=np.float16):

        if task is None:
            model_path = os.path.join(model_dir, 'weights.npz')
            if os.path.exists(model_path):
                npz = np.load(model_path)
                sizes = npz['sizes']
                bi_ratio = np.array(npz['bi_ratio'], dtype=dtype)

                node_weight = npz['node_weight'].astype(dtype)  # 强制转换 数据类型
                edge_weight = npz['edge_weight'].astype(dtype)  # 强制转换 数据类型

                model = cls.__new__(cls)
                model.n_tag = int(sizes[0])
                model.n_feature = int(sizes[1])
                model.bi_ratio = bi_ratio
                model.node_weight = node_weight
                model.edge_weight = edge_weight

                assert model.node_weight.shape[0] == model.n_feature
                assert model.node_weight.shape[1] == model.n_tag
                assert model.edge_weight.shape[0] == model.n_tag
                assert model.edge_weight.shape[1] == model.n_tag

                return model

            raise FileNotFoundError('the model file `{}` does not exist.'.format(model_path))

        elif task == 'cws':
            model_path = os.path.join(model_dir, 'weights.npz')
            if os.path.exists(model_path):
                npz = np.load(model_path)
                sizes = npz['sizes']
                bi_ratio = np.array(npz['bi_ratio'], dtype=dtype)

                node_weight = npz['node_weight'].astype(dtype)  # 强制转换 数据类型
                edge_weight = npz['edge_weight'].astype(dtype)  # 强制转换 数据类型

                node_weight_expand = np.expand_dims(node_weight, axis=1)
                node_weight = np.concatenate((node_weight_expand, - node_weight_expand), axis=1)

                # 整理参数集，节约存储空间
                opposite_diff_path = os.path.join(model_dir, 'opposite_diff.txt')
                if os.path.exists(opposite_diff_path):
                    with open(opposite_diff_path, 'r', encoding='utf-8') as fr:

                        for line in fr.readlines():
                            key, value = line.strip().split('\t')
                            node_weight[int(key)][1] = float(value)
                            # pdb.set_trace()

                model = cls.__new__(cls)
                model.n_tag = int(sizes[0])
                model.n_feature = int(sizes[1])
                model.bi_ratio = bi_ratio
                model.node_weight = node_weight
                model.edge_weight = edge_weight

                assert model.node_weight.shape[0] == model.n_feature
                assert model.node_weight.shape[1] == model.n_tag
                assert model.edge_weight.shape[0] == model.n_tag
                assert model.edge_weight.shape[1] == model.n_tag

                return model

            raise FileNotFoundError('the model file `{}` does not exist.'.format(model_path))

    def save(self):
        if self.task is None:
            sizes = np.array([self.n_tag, self.n_feature])
            np.savez(os.path.join(self.config.model_dir, 'weights.npz'),
                sizes=sizes, bi_ratio=self.bi_ratio,
                node_weight=self.node_weight, edge_weight=self.edge_weight)

        elif self.task == 'cws':
            # 重新调整 node_weight 的压缩
            weight_gap = 0.0001  # 该权重指 np.float16 可分辨最小精度，故可默认选取
            opposite_diff_dict = dict()
            for idx in range(self.node_weight.shape[0]):
                tmp_node_weight = self.node_weight[idx]
                if abs(tmp_node_weight[0] + tmp_node_weight[1]) > weight_gap:
                    # format 格式化将造成某些权重小于万分位的出错，但几乎不影响模型性能
                    tmp_str = '{:.4f}'.format(float(tmp_node_weight[1]))
                    opposite_diff_dict.update({idx: tmp_str})

            new_node_weight = np.array(self.node_weight[:, 0], dtype=np.float16)  # 默认的压缩精度
            new_edge_weight = np.array(self.edge_weight, dtype=np.float16)
            logging.info('opposite num: {}, percent: {:.3%}'.format(
                len(opposite_diff_dict), len(opposite_diff_dict) / self.node_weight.shape[0]))

            sizes = np.array([self.n_tag, self.n_feature])
            np.savez(os.path.join(self.config.model_dir, 'weights.npz'),
                sizes=sizes, bi_ratio=self.bi_ratio,
                node_weight=new_node_weight, edge_weight=new_edge_weight)

            # np.savez(os.path.join(self.config.model_dir, 'orig_weights.npz'),
            #     sizes=sizes, bi_ratio=self.bi_ratio,
            #     node_weight=self.node_weight, edge_weight=self.edge_weight)

            # 写入新文件 opposite_diff.txt
            with open(os.path.join(self.config.model_dir, 'opposite_diff.txt'),
                      'w', encoding='utf-8') as fw:
                for key, value in opposite_diff_dict.items():
                    fw.write(str(key) + '\t' + value + '\n')
