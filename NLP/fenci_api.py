#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Using gensim handle the chinese text.
"""

from gensim.models import word2vec


def calculate_dis(word1, word2):
    model = word2vec.Word2Vec.load("zh.model")
    return model.similarity(word1, word2)

if __name__ == "__main__":
    word_1 = u'鸡'
    word_2 = u'尾'
    print calculate_dis(word_1, word_2)
