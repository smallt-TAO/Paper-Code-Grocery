#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：测试gensim使用
时间：2016年5月21日 18:07:50
"""

from gensim.models import word2vec
import logging

# 主程序
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 对应的加载方式
model_2 = word2vec.Word2Vec.load("text8.model")

# 计算两个词的相似度/相关程度
y1 = model_2.similarity("woman", "man")
print u"woman和man的相似度为：", y1
print "--------\n"

# 计算某个词的相关词列表
y2 = model_2.most_similar("good", topn=20)  # 20个最相关的
print u"和good最相关的词有：\n"
for item in y2:
    print item[0], item[1]
print "--------\n"
