# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'
# Website: http://www.jionlp.com/


import os
import pdb
import time
import random
import numpy as np
from multiprocessing import Process, Queue

from jiojio import TimeIt, logging

from jiojio.dataset import DataSet
from jiojio.cws.feature_extractor import CWSFeatureExtractor
from jiojio.cws.scorer import F1_score

from jiojio.model import Model
from jiojio.inference import decodeViterbi_fast
from jiojio.optimizer import SGD


def train(config):

    feature_extractor = CWSFeatureExtractor(config)

    if config.build_train_temp_files:
        # 构建 特征数据集
        with TimeIt('# build datasets'):
            feature_extractor.build(config.train_file)
            # feature_extractor.save()

        with TimeIt('# make feature files'):
            feature_extractor.convert_text_file_to_feature_idx_file(
                config.train_file, config.feature_train_file, config.gold_train_file)
            feature_extractor.convert_text_file_to_feature_idx_file(
                config.test_file, config.feature_test_file, config.gold_test_file)

    else:
        # 否则直接加载内置的已预处理的数据集文件
        feature_extractor = feature_extractor.load(config, config.model_dir)

    logging.info("\nstart training ...")

    with TimeIt('loading dataset'):
        train_set = DataSet.load(config.feature_train_file, config.gold_train_file)
        logging.info("train_set size: {}\n".format(len(train_set)))

    trainer = Trainer(config, train_set, feature_extractor)

    for i in range(config.train_epoch):
        logging.info('- epoch {}:'.format(i))
        with TimeIt('training epoch {}'.format(i)):
            err, diff = trainer.train_epoch()

        # 测试
        with TimeIt('loading valid dataset'):
            if i == config.train_epoch - 1:  # 最后一个 epoch 用全量
                sample_ratio = 1
            else:
                sample_ratio = config.sample_ratio  # 仅用全数据量的少量数据做训练中验证

            train_valid_set = DataSet.load(
                config.feature_train_file, config.gold_train_file, sample_ratio=config.sample_ratio)
            test_valid_set = DataSet.load(
                config.feature_test_file, config.gold_test_file, sample_ratio=sample_ratio)

        with TimeIt('Testing'):
            logging.info('# train_set:')
            train_score_list = trainer.test(train_valid_set)
            logging.info('# test_set:')
            test_score_list = trainer.test(test_valid_set)

            # 计算所有参数的最大值，平均值，中位值，确保模型的参数稳定
            max_weight = np.max(trainer.model.node_weight)
            min_weight = np.min(trainer.model.node_weight)
            average_weight = np.sum(trainer.model.node_weight) / len(trainer.model.node_weight)
            average_abs_weight = np.sum(np.abs(trainer.model.node_weight)) / len(trainer.model.node_weight)
        # pdb.set_trace()
        logging.info(
            "- epoch {}: \n"
            "\t- diff={:.4f}  error={:.4f}\n"
            "\t- max-weight={:.4f}  min-weight={:.4f}\n"
            "\t  average-weight={:.4f}  average-abs-weight={:.4f}".format(
                i, diff, err, max_weight, min_weight,
                average_weight, average_abs_weight))

        logging.info('saving the epoch model.')
        feature_extractor.save()
        trainer.model.save()
        logging.info('-' * 50 + '\n')

    # 重新整理参数，将那些不生效的参数剔除，例如，node_score 中，B 和 I 的值几乎相同，且远低于特征的平均值
    '''  # 可删除的参数过少，可以忽略，仅占 0.018%
    new_node_weight, new_feature_to_idx = params_cut(
        trainer.model.node_weight, feature_extractor.feature_to_idx)
    trainer.model.node_weight = new_node_weight
    feature_extractor.feature_to_idx = new_feature_to_idx
    trainer.model.n_feature = new_node_weight.shape[0]
    '''
    # feature_extractor.save()
    # trainer.model.save()
    config.to_json()

    logging.info("finished.")


def params_cut(node_weight, feature_to_idx):
    # 删除哪些权重区分度极低的参数
    idx_to_feature = dict([(value, key) for key, value in feature_to_idx.items()])

    weight_gap = 1e-4  # 由 np.float16 的最小分辨精度决定
    # weight_max_mean = node_weight.max(axis=1).mean()

    new_feature_list = list()
    new_node_weight_list = list()
    params_being_cut = list()
    pdb.set_trace()
    for idx in range(node_weight.shape[0]):
        if node_weight[idx].max() - node_weight[idx].min() <= weight_gap:
            # 权重分布均匀，无较大的区分度
            params_being_cut.append(idx_to_feature[idx])
            continue

        new_feature_list.append(idx_to_feature[idx])
        new_node_weight_list.append(idx)

    logging.info('cut {} params from total {}, cut ratio {:.6%}.'.format(
        len(params_being_cut), node_weight.shape[0],
        len(params_being_cut) / node_weight.shape[0]))

    random.shuffle(params_being_cut)
    logging.info(', '.join(params_being_cut[:100]))

    new_node_weight = node_weight[new_node_weight_list]
    new_feature_to_idx = dict([(key, idx) for idx, key in enumerate(new_feature_list)])
    assert new_node_weight.shape[0] == len(new_feature_to_idx)

    return new_node_weight, new_feature_to_idx


class Trainer(object):

    def __init__(self, config, dataset, feature_extractor):
        self.config = config
        self.n_feature = dataset.n_feature
        self.n_tag = dataset.n_tag

        self.model = Model(config, self.n_feature, self.n_tag, task='cws')

        self.optim = self._get_optimizer(dataset, self.model)

        self.feature_extractor = feature_extractor
        self.idx_to_chunk_tag = dict()
        for tag, idx in feature_extractor.tag_to_idx.items():
            # if tag.startswith("I"):
            #     tag = "I"
            self.idx_to_chunk_tag[idx] = tag

    def _get_optimizer(self, dataset, model):
        config = self.config
        return SGD(config, dataset, model)

    def train_epoch(self):
        return self.optim.optimize()

    def train_edge_params(self):
        # 即直接根据语料统计参数取值
        samples_num = len(self.optim.dataset)
        tag_length = len(self.feature_extractor.tag_to_idx)
        edge_count_matrix = np.zeros((tag_length, tag_length), dtype=np.int32)

        for sample in self.optim.dataset:
            sample.tags  # 未完
        return

    def _decode(self, test_set: DataSet, model: Model):
        if self.config.nThread == 1:
            self._decode_single(test_set, model)
        else:
            self._decode_multi_proc(test_set, model)

    def _decode_single(self, test_set: DataSet, model: Model):
        for example in test_set:
            example.features = [list(map(int, feature_line.split(",")))
                                for feature_line in example.features.split("\n")]
            example.tags = list(map(int, example.tags.split(',')))
            tags = decodeViterbi_fast(example.features, model)
            example.predicted_tags = tags
            example.features = None

    @staticmethod
    def _decode_proc(model, in_queue, out_queue):
        while True:
            item = in_queue.get()
            if item is None:
                return

            idx, features = item
            tags = decodeViterbi_fast(features, model)
            out_queue.put((idx, tags))

    def _decode_multi_proc(self, test_set: DataSet, model: Model):
        in_queue = Queue()
        out_queue = Queue()
        procs = list()
        nthread = self.config.nThread
        for i in range(nthread):
            p = Process(
                target=self._decode_proc, args=(model, in_queue, out_queue))
            procs.append(p)

        for idx, example in enumerate(test_set):
            example.features = [list(map(int, feature_line.split(",")))
                                for feature_line in example.features.split("\n")]
            in_queue.put((idx, example.features))

        for proc in procs:
            in_queue.put(None)
            proc.start()

        for _ in range(len(test_set)):
            idx, tags = out_queue.get()
            test_set[idx].predicted_tags = tags

        for p in procs:
            p.join()

    def test(self, dataset):
        self._decode(dataset, self.model)

        gold_tags = list()
        pred_tags = list()

        sample_wrong = 0
        token_wrong = 0
        token_total = 0
        for example in dataset:
            pred_tags.append(example.predicted_tags)
            example.tags = list(map(int, example.tags.split(',')))
            gold_tags.append(example.tags)

            flag = False
            for pred_tag, gold_tag in zip(example.predicted_tags, example.tags):
                if pred_tag != gold_tag:
                    flag = True
                    token_wrong += 1

            token_total += len(example.tags)
            if flag:
                sample_wrong += 1  # 样本错误

        score_list, info_list = F1_score(gold_tags, pred_tags, self.idx_to_chunk_tag)
        logging.info("\t- test-sample-num={}\n"
                     "\t- gold-num={}  output-num={}  correct-num={}\n"
                     "\t- precision={:.2%}  recall={:.2%}  f-score={:.2%}\n"
                     "\t- token_acc={:.2%}  sample_acc={:.2%}\n".format(
                         len(dataset), info_list[0], info_list[1], info_list[2],
                         score_list[1], score_list[2], score_list[0],
                         (token_total - token_wrong) / token_total,
                         (len(dataset) - sample_wrong) / len(dataset)))

        return score_list
