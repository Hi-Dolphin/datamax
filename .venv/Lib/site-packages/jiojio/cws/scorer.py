# -*- coding=utf-8 -*-
# Library: jiojio
# Author: dongrixinyu
# License: GPL-3.0
# Email: dongrixinyu.89@163.com
# Github: https://github.com/dongrixinyu/jiojio
# Description: fast Chinese Word Segmentation(CWS) and Part of Speech(POS) based on CPU.'

import pdb


def F1_score(gold_tags_list, pred_tags_list, idx_to_chunk_tag):
    """ 计算 序列标注的 F1 值 """
    assert len(pred_tags_list) == len(gold_tags_list)

    gold_tags_list = tag_idx_2_token(idx_to_chunk_tag, gold_tags_list)
    pred_tags_list = tag_idx_2_token(idx_to_chunk_tag, pred_tags_list)

    gold_chunk_list = get_chunks(gold_tags_list)
    pred_chunk_list = get_chunks(pred_tags_list)

    gold_chunk_num = 0
    pred_chunk_num = 0
    correct_chunk_num = 0
    for i in range(len(gold_chunk_list)):
        gold_chunk_num += len(gold_chunk_list[i])
        pred_chunk_num += len(pred_chunk_list[i])

        for im in pred_chunk_list[i]:
            if im in gold_chunk_list[i]:
                correct_chunk_num += 1

    precision = correct_chunk_num / pred_chunk_num
    recall = correct_chunk_num / gold_chunk_num
    f1 = 0 if correct_chunk_num == 0 else 2 * precision * recall / (precision + recall)

    score_list = [f1, precision, recall]
    info_list = [gold_chunk_num, pred_chunk_num, correct_chunk_num]

    return score_list, info_list


def tag_idx_2_token(tag_map, tags_list):
    '''
    res_list = list()
    for tags in tags_list:
        sample_list = list()
        for tag in tags:
            sample_list.append(tag_map[tag])
        res_list.append(sample_list)

    return res_list
    '''
    return [[tag_map[tag] for tag in tags] for tags in tags_list]


def get_chunks(tags_list):
    chunks_list = list()
    for tags in tags_list:
        chunks = list()
        for i in range(len(tags)):
            if tags[i] == 'B':
                pos = i
                length = 1
                for j in range(i + 1, len(tags)):
                    if tags[j] == 'I':
                        length += 1
                    else:
                        break

                chunks.append(str(length) + "*" + str(pos))
        chunks_list.append(chunks)

    return chunks_list
