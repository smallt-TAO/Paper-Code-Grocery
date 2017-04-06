#! -*- coding:utf-8 -*-
import string
import jieba
seg_list = jieba.cut("我来到北京清华大学", cut_all = True)
print "Full Mode:", ' '.join(seg_list)

seg_list = jieba.cut("我来到北京清华大学")
print "Default Mode:", ' '.join(seg_list)

__dict = {}
def load_dict(dict_file='words.dic'):
    #加载词库，把词库加载成一个key为首字符，value为相关词的列表的字典
   
    words = [unicode(line, 'utf-8').split() for line in open(dict_file)]
   
    for word_len, word in words:
        first_char = word[0]
        __dict.setdefault(first_char, [])
        __dict[first_char].append(word)
      
    #按词的长度倒序排列
    for first_char, words in __dict.items():
        __dict[first_char] = sorted(words, key=lambda x:len(x), reverse=True)
