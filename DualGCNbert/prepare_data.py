# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:04:53 2023

@author: 86189
"""

import json
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset
import sys
import torch
sys.path.append('../LAL_Parser/src_joint')
#%%
def ParseData(data_path):
    initial = 'initial' in data_path
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        j = 0
        for _, d in enumerate(data):
            # different handling of sentence_id for 'initial' and others
            if not initial:
                sentence_id = d['sentence_id']
            for aspect in d['aspects']:
                j += 1
                if initial:
                    sentence_id = j
                text_list = list(d['token'])
                tok = list(d['token'])       # word token
                length = len(tok)            # real length
                if length < 150:
                    tok = [t.lower() for t in tok]
                    tok = ' '.join(tok)
                    try:
                        asp = list(aspect['aspect'])  
                    except:
                        asp = list(aspect['term']) # aspect
                    asp = [a.lower() for a in asp]
                    asp = ' '.join(asp)
                    label = aspect['polarity']   # label
                    aspect_post = [aspect['from'], aspect['to']] 
                    post = [i-aspect['from'] for i in range(aspect['from'])] \
                       +[0 for _ in range(aspect['from'], aspect['to'])] \
                       +[i-aspect['to']+1 for i in range(aspect['to'], length)]
                    if len(asp) == 0:
                        mask = [1 for _ in range(length)]    # for rest16
                    else:
                        mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]
                
                    sample = {'text': tok, 'aspect': asp, 'post': post, \
                           'length': length, 'label': label, 'mask': mask, \
                          'aspect_post': aspect_post, 'text_list': text_list, \
                          'sentence_id': sentence_id}  # added sentence_id
                    all_data.append(sample)
    return all_data


def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x

class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    #然后用BERT模型的分词器把每个token转化为一个唯一的ID。转化后的ID列表返回。
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
class DualGCNBertData(Dataset):
    # 初始化函数，负责读取数据，处理数据，并保存到self.data中。
    def __init__(self, fname, tokenizer, opt):
        self.data = []   # 初始化数据存储列表
        parse = ParseData 
        polarity_dict = {'positive':2, 'negative':0, 'neutral':1}  # 定义情感标签到整数的映射
        
        # 遍历数据集中的每一条数据
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            if obj['label'] is not None:
                polarity = polarity_dict[obj['label']]
            else:
                polarity = 3   # 获取情感标签，并转换为整数
            text = obj['text']   # 获取句子
            term = obj['aspect']   # 获取句子中的aspect
            term_ = ''.join(term)
            sentence_id = obj['sentence_id']
            term_start = obj['aspect_post'][0]   # 获取aspect在句子中的开始位置
            term_end = obj['aspect_post'][1]   # 获取aspect在句子中的结束位置
            text_list = obj['text_list']  # 获取句子的列表形式
            # 将句子按照aspect切分为左半部分，aspect部分，右半部分
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end: ]
            # 下面的代码负责依存句法分析，并构建依存句法矩阵
            # ori_adj: the dependency graph of the sentence
            def contains_nan_or_inf(arr):
                return np.isnan(arr).any() or np.isinf(arr).any()
            from absa_parser import headparser
            try:
                headp, syntree = headparser.parse_heads(text)
            except:
                print(f"Ran out of CUDA memory at sentence_id {obj['sentence_id']} or shape error")
                torch.cuda.empty_cache()
                continue
            ori_adj = softmax(headp[0])
            ori_adj = np.delete(ori_adj, 0, axis=0)
            ori_adj = np.delete(ori_adj, 0, axis=1)
            ori_adj -= np.diag(np.diag(ori_adj))
            if contains_nan_or_inf(ori_adj):  # Check for nan or inf values
                print(f"Found NaN or Inf values at sentence_id {obj['sentence_id']}")
                continue
            if not opt.direct:
                ori_adj = ori_adj + ori_adj.T
            ori_adj = ori_adj + np.eye(ori_adj.shape[0])
            try:
                assert len(text_list) == ori_adj.shape[0] == ori_adj.shape[1], '{}-{}-{}'.format(len(text_list), text_list, ori_adj.shape)

            # 对左半部分，aspect部分，右半部分进行分词，并建立分词到原始词的映射
            # 对左半部分进行处理
                left_tokens, term_tokens, right_tokens = [], [], []
                left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []
                for ori_i, w in enumerate(left):
                    for t in tokenizer.tokenize(w):
                        left_tokens.append(t)
                        left_tok2ori_map.append(ori_i)
                asp_start = len(left_tokens)
                offset = len(left) 

            # 对aspect部分进行处理
                for ori_i, w in enumerate(term):
                    for t in tokenizer.tokenize(w):
                        term_tokens.append(t)
                        term_tok2ori_map.append(ori_i + offset)
                asp_end = asp_start + len(term_tokens)
                offset += len(term)

            # 对右半部分进行处理
                for ori_i, w in enumerate(right):
                    for t in tokenizer.tokenize(w):
                        right_tokens.append(t)
                        right_tok2ori_map.append(ori_i+offset)

            # 对过长的句子进行截断
                while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len-2*len(term_tokens) - 3:
                    if len(left_tokens) > len(right_tokens):
                        left_tokens.pop(0)
                        left_tok2ori_map.pop(0)
                        asp_start -= 1
                        asp_end -= 1
                    else:
                        right_tokens.pop()
                        right_tok2ori_map.pop()

            # 对截断后的句子进行编码，并创建对应的依存句法矩阵
                bert_tokens = left_tokens + term_tokens + right_tokens
                tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
                truncate_tok_len = len(bert_tokens)
                tok_adj = np.zeros((truncate_tok_len, truncate_tok_len), dtype='float32')
                for i in range(truncate_tok_len):
                    for j in range(truncate_tok_len):
                        tok_adj[i][j] = ori_adj[tok2ori_map[i]][tok2ori_map[j]]

            # 对句子进行编码，包括将词转换为ID，添加CLS和SEP等特殊标记
                context_asp_ids = [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(
                    bert_tokens)+[tokenizer.sep_token_id]+tokenizer.convert_tokens_to_ids(term_tokens)+[tokenizer.sep_token_id]
                context_asp_len = len(context_asp_ids)
                paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
                context_len = len(bert_tokens)
                context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
                src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
                aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
                aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
                context_asp_attention_mask = [1] * context_asp_len + paddings
                context_asp_ids += paddings
                context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
                context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
                context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
                src_mask = np.asarray(src_mask, dtype='int64')
                aspect_mask = np.asarray(aspect_mask, dtype='int64')
            # 创建依存句法矩阵
                context_asp_adj_matrix = np.zeros((tokenizer.max_seq_len, tokenizer.max_seq_len)).astype('float32')
                context_asp_adj_matrix[1:context_len + 1, 1:context_len + 1] = tok_adj
            # 创建一个字典，存储所有的数据，并添加到self.data中
                data = {
                    'text': text,
                    'term': term_,
                    'asp_start': asp_start,
                    'asp_end': asp_end,
                    'text_bert_indices': context_asp_ids,
                    'bert_segments_ids': context_asp_seg_ids,
                    'attention_mask':context_asp_attention_mask,
                    'adj_matrix': context_asp_adj_matrix,
                    'src_mask': src_mask,
                    'aspect_mask': aspect_mask,
                    'polarity': polarity,
                    'sentence_id': sentence_id
                }
                self.data.append(data)
            except AssertionError:
                print(f"Skipped data for sentence id: {sentence_id} due to assertion error.")
                continue
    # 返回数据集大小
    def __len__(self):
        return len(self.data)

    # 根据索引返回一条数据
    def __getitem__(self, idx):
        return self.data[idx]

    def append(self, sample):
        self.data.append(sample)
