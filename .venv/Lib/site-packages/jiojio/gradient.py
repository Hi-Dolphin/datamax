# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'

import pdb
import copy
import time
import numpy as np
from typing import List
import multiprocessing as mp

from jiojio.dataset import Sample
from jiojio.model import Model
from jiojio.inference import get_Y_YY, get_beliefs, bi_ratio_loss, \
    get_masked_beliefs, Belief, MaskedBelief


def combine_two_dict(dict_1, dict_2):
    # 将两个统计 dict 合二为一

    for key in dict_2:
        if key in dict_1:
            dict_1[key] += dict_2[key]

        else:
            dict_1.update({key: dict_2[key]})

    return dict_1


def get_grad_SGD_minibatch(node_grad: np.ndarray, edge_grad: np.ndarray,
                           model: Model, X: List[Sample], process_num=4):
    feature_id_dict = dict()
    errors = 0
    total_num = 0
    total_with_edge_num = 0
    total_with_delta_edge_num = 0

    # 多进程处理
    pool = mp.Pool(processes=process_num)
    sample_idx = 0
    jump_step = (len(X) // process_num) + 1
    result_list = list()
    for p_n in range(process_num):

        sub_X = X[p_n * jump_step: (p_n + 1) * jump_step]
        sub_node_grad = copy.deepcopy(node_grad)
        sub_edge_grad = copy.deepcopy(edge_grad)

        p_res = pool.apply_async(
            func=get_grad_SGD_minibatch_subprocess,
            args=(sub_node_grad, sub_edge_grad, model, sub_X))
        # print('start subprocess ...')
        result_list.append(p_res)

    # 融合各子进程得到的数据
    for result in result_list:
        error, sub_feature_id_dict, sub_total_num, sub_total_with_edge_num, \
            sub_total_with_delta_edge_num, sub_node_grad, sub_edge_grad = result.get()

        errors += error
        feature_id_dict = combine_two_dict(feature_id_dict, sub_feature_id_dict)
        total_num += sub_total_num
        total_with_edge_num += sub_total_with_edge_num
        total_with_delta_edge_num += sub_total_with_delta_edge_num

        node_grad += sub_node_grad
        edge_grad += sub_edge_grad
        # print(edge_grad)
        # pdb.set_trace()
    pool.close()
    pool.join()

    for feature_id, num in feature_id_dict.items():
        node_grad[feature_id] /= num

    # 由于梯度值是基于 bi_ratio 为 1 base 计算，因此，将edge_grad 的变化幅度减小 model.bi_raito 倍
    # edge_grad *= model.bi_ratio
    edge_grad /= len(X)

    # print('total: {}, with_edge_num: {}, with_delta_edge_num: {}.'.format(
    #     total_num, total_with_edge_num, total_with_delta_edge_num))

    # pdb.set_trace()
    # 防止梯度出现频繁的变动，尤其是在前期
    # 但 1e-4 的值将导致在 sample_num 较小时失效
    if np.abs(total_with_edge_num - total_with_delta_edge_num) < np.ceil(total_num * 2e-5):  # 不频繁变动
        bi_ratio_grad = 0
    else:
        bi_ratio_grad = (total_with_edge_num - total_with_delta_edge_num) / total_num
    # pdb.set_trace()
    return errors / len(X), feature_id_dict, bi_ratio_grad


def get_grad_SGD_minibatch_subprocess(
        node_grad: np.ndarray, edge_grad: np.ndarray,
        model: Model, X: List[Sample]):

    feature_id_dict = dict()
    errors = 0
    total_num = 0
    total_with_edge_num = 0
    total_with_delta_edge_num = 0
    for x in X:
        # start_time = time.time()
        # pdb.set_trace()
        error, feature_id_set, node_num, with_edge_num, with_delta_edge_num = get_grad_CRF(
            node_grad, edge_grad, model, x)

        total_num += node_num
        total_with_edge_num += with_edge_num
        total_with_delta_edge_num += with_delta_edge_num

        # print('time: {:.2f}, len: {}, ratio: {:.2f}'.format(
        #     time.time() - start_time, len(x.features.split('\n')), len(x.features.split('\n')) / (time.time() - start_time)))
        errors += error

        for i in feature_id_set:
            if i in feature_id_dict:
                feature_id_dict[i] += 1
            else:
                feature_id_dict.update({i: 1})

    return errors, feature_id_dict, total_num, \
        total_with_edge_num, total_with_delta_edge_num, node_grad, edge_grad


def get_grad_CRF(node_grad: np.ndarray, edge_grad: np.ndarray,
                 model: Model, x: Sample):
    feature_id_set = set()

    n_tag = model.n_tag

    # 为节省内存，每次均须将字符串解为 list 处理，增加了耗时
    orig_features = x.features
    orig_tags = x.tags
    x.features = [list(map(int, feature_line.split(",")))
                  for feature_line in orig_features.split("\n")]
    x.tags = list(map(int, orig_tags.split(',')))

    belief = Belief(len(x), n_tag)
    belief_masked = MaskedBelief(len(x), n_tag)

    Y, YY, masked_Y = get_Y_YY(model, x)

    Z, sum_edge = get_beliefs(belief, Y, YY, model.bi_ratio)
    sum_edge_masked = get_masked_beliefs(belief_masked, masked_Y)

    for i, node_feature_list in enumerate(x.features):
        diff = belief.node_states[i] - belief_masked.node_states[i]
        for feature_id in node_feature_list:

            feature_id_set.add(feature_id)  # 需要更新梯度值的特征值索引
            node_grad[feature_id] += diff  # 计算梯度

    # 处理 bi_ratio 参数
    node_num, with_edge_correct_num, without_edge_correct_num = bi_ratio_loss(
        Y, belief_masked.node_states, YY, bi_ratio=model.bi_ratio)

    # 更新转移概率梯度
    edge_grad += np.reshape(sum_edge - sum_edge_masked, (n_tag, n_tag))

    # pdb.set_trace()
    #还原数据，节省内存
    x.features = orig_features
    x.tags = orig_tags

    return Z, feature_id_set, node_num, with_edge_correct_num, without_edge_correct_num
