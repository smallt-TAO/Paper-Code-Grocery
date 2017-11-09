#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：测试gensim使用，处理中文语料
时间：2016年5月21日 20:49:07
"""

from gensim.models import word2vec
import logging

# 主程序
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus(u"all_zi.txt")  # 加载语料
model = word2vec.Word2Vec(sentences, size=200)  # 训练skip-gram模型; 默认window=5

# 保存模型，以便重用
model.save(u"zh.model")
# 对应的加载方式
# model_2 = word2vec.Word2Vec.load("text8.model")


if __name__ == "__main__":
    pass
