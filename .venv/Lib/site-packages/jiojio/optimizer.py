# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'

import pdb
import random

import numpy as np
from jiojio import logging
from jiojio.gradient import get_grad_SGD_minibatch


class Optimizer(object):

    def __init__(self):
        self._pre_values = list()

    def converge_test(self, err):
        val = err
        if len(self._pre_values) > 0:
            if len(self._pre_values) == 10:  # 超出长度，弹出第一个值
                self._pre_values.pop(0)

            average_improvement = (self._pre_values[0] - err) / \
                len(self._pre_values)  # 训练提升量
            val = average_improvement / abs(err)  # 训练提升比率

        self._pre_values.append(err)
        return val

    def optimize(self):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, config, dataset, model):
        super().__init__()

        self.config = config

        self._model = model
        self.dataset = dataset
        self.learning_rate = config.initial_learning_rate
        self.training_epoch_num = 0  # 计算训练轮数

    def _tune_reshuffle_samples(self):
        # 调整样本，打乱样本
        sample_num = len(self.dataset)

        random_index = list(range(sample_num))
        random.shuffle(random_index)  # 样本打乱
        remainder = sample_num % self.config.mini_batch
        if remainder != 0:
            sample_num += (self.config.mini_batch - remainder)
            random_index.extend(random_index[:(self.config.mini_batch - remainder)])

        return sample_num, random_index

    def optimize(self):
        logging.info('model w init: {:.6f} ... {:.6f}'.format(
            self._model.node_weight[0][0], self._model.edge_weight[0][0]))

        error_list = list()

        max_weight_num = 10
        max_step_num = 1
        sample_num, random_index = self._tune_reshuffle_samples()

        for t in range(0, sample_num, self.config.mini_batch):
            node_grad = np.zeros((self._model.n_feature, self._model.n_tag), dtype=np.float32)  # 节点梯度值
            edge_grad = np.zeros((self._model.n_tag, self._model.n_tag), dtype=np.float32)  # 转移梯度值
            XX = list()
            for i in random_index[t: t + self.config.mini_batch]:
                XX.append(self.dataset[i])  # 小 batch 样本

            error, feature_set, bi_ratio_grad = get_grad_SGD_minibatch(
                node_grad, edge_grad, self._model, XX, process_num=self.config.process_num)
            error_list.append(error)

            # update decay rates
            self.learning_rate = self.config.initial_learning_rate * \
                np.exp((- self.training_epoch_num - t / sample_num) * self.config.dropping_rate)
            if t % (self.config.mini_batch * self.config.interval) == 0:
                logging.info('\tlr: {:.5f}, sample idx {}: grad: {:.6f} ... {:.6f}'.format(
                    self.learning_rate, t, node_grad[0][0], edge_grad[0][0]))

            # update weights
            # 若学习率与步长的乘积大于某个数值，则说明梯度爆炸，应该进行规约
            node_delta = self.learning_rate * node_grad
            edge_delta = self.learning_rate * edge_grad
            bi_ratio_grad = bi_ratio_grad / (0.01 * self._model.bi_ratio)  # 真正的梯度
            bi_ratio_delta = bi_ratio_grad * self.learning_rate
            node_delta[node_delta > max_step_num] = max_step_num
            node_delta[node_delta < -max_step_num] = -max_step_num
            # if t % (self.config.mini_batch * self.config.interval) == 0:
            #     print('grad:   ', node_delta[0][:5])
            #     print('weight: ', self._model.node_weight[0][:5])

            self._model.bi_ratio += bi_ratio_delta
            if self._model.bi_ratio < 0:
                self._model.bi_ratio = 1e-6
            self._model.bi_ratio = np.array(self._model.bi_ratio, dtype=np.float32)

            if t % (self.config.mini_batch * self.config.interval) == 0:
                logging.info('current bi_ratio: {:.6f}, delta: {:.4f}, max edge_w: {:.4f}'.format(
                             self._model.bi_ratio, bi_ratio_delta, np.max(self._model.edge_weight)))

            self._model.node_weight -= node_delta
            self._model.edge_weight -= edge_delta

            # print('max edge_weight: {:.3f} max node_weight: {:.3f}'.format(
            #     np.max(np.abs(self._model.edge_weight)),
            #     np.max(np.abs(self._model.node_weight))))
            self._model.edge_weight[self._model.edge_weight > max_weight_num] = max_weight_num
            self._model.edge_weight[self._model.edge_weight < -max_weight_num] = -max_weight_num
            self._model.node_weight[self._model.node_weight > max_weight_num] = max_weight_num
            self._model.node_weight[self._model.node_weight < -max_weight_num] = -max_weight_num

            if self.config.regularization:
                # 参数正则化，该公式，对大参数值的惩罚越大
                node_r2 = self.learning_rate * self._model.node_weight
                edge_r2 = self.learning_rate * self._model.edge_weight

                self._model.node_weight -= node_r2
                self._model.edge_weight -= edge_r2

        diff = self.converge_test(sum(error_list) / len(error_list))

        logging.info('err/diff: {:.4f}/{:.4f}'.format(sum(error_list) / len(error_list), diff))
        logging.info('model w change: {:.6f} ... {:.6f}'.format(
            self._model.node_weight[0][0], self._model.edge_weight[0][0]))

        self.training_epoch_num += 1

        return error, diff
