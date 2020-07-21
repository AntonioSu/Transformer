# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp

import numpy as np
import codecs
import re
import random
import torch
import jieba


def load_chi_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/chi.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/eng.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents):
    en2idx, idx2en = load_en_vocab()
    chi2idx, idx2chi = load_chi_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [en2idx.get(word, 1) for word in (source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        target_list=jieba.cut(target_sent)
        y = [chi2idx.get(word, 1) for word in target_list]
        y=y+[3] #add the </S>
        if max(len(x), len(y)) <=hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets
def load_train_data():
    en_sents = [line.strip() for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n")]
    de_sents = [line.strip() for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n")]

    X, Y, Sources, Targets = create_data(en_sents, de_sents)
    return X, Y
    
def load_test_data():
    de_sents = [line.strip() for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") ]
    en_sents = [line.strip() for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") ]
        
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets # (1064, 150)


def get_batch_indices(total_length, batch_size):
    current_index = 0
    indexs = [i for i in range(total_length)]
    random.shuffle(indexs)
    while 1:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexs[current_index: current_index + batch_size], current_index


