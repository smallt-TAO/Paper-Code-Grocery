#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Using gensim handle the chinese text.
"""

from gensim.models import word2vec


def calculate_dis(word1, word2):
    model = word2vec.Word2Vec.load("zh.model")
    return model.similarity(word1, word2)


def fen_ci(sen, threshold=0.013):
    wor, res, se = [], [], ""
    for s in sen:
        wor.append(s)
    for i in range(1, len(wor)):
        if calculate_dis(wor[i - 1], wor[i]) > threshold:
            res.append(wor[i - 1])
        else:
            res.append(wor[i - 1])
            res.append(" ")
    res.append(wor[-1])
    for i in res:
        se += i
    print se


if __name__ == "__main__":
    word_1 = u'厉'
    word_2 = u'害'
    print calculate_dis(word_1, word_2)
    input_word = u"蛊毒最厉害"
    fen_ci(input_word)
    input_word = u"人民医院"
    fen_ci(input_word, threshold=0.2)
