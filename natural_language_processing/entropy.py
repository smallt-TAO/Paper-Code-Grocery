# -*- coding: utf-8 -*-

import re
import jieba
import codecs
import collections
import math


def deal_file(file1):
    output_file = codecs.open(file1, mode='w', encoding='utf-8')

    text1seg = []
    for i in range(10):
        file1 = '.\\ebook\\{}.txt'.format(str(i + 1))
        print str(file1)
        input_file = codecs.open(file1, mode='r', encoding='utf-8')

        for line in input_file.readlines():
            line_seg = jieba.cut(line)
            text1seg.extend(line_seg)
        input_file.close()

    for word in text1seg:
        output_file.write(word + ' ')
    output_file.close()


def save_vocab(file1, file2):
    text1seg = []
    input_file = codecs.open(file1, mode='r', encoding='utf-8')
    output = codecs.open(file2, mode='w', encoding='utf-8')

    for line in input_file.readlines():
        text1seg.extend(line.split())

    vocab = collections.Counter(text1seg)
    total_num = len(text1seg)
    output.write("{}".format(total_num))
    output.write("\n")
    for key in vocab:
        output.write(key + "  ")
        output.write("{}".format(vocab[key]))
        output.write('\n')

    input_file.close()
    output.close()


def sort_by_count(d):
    # 字典排序
    d = collections.OrderedDict(sorted(d.items(), key=lambda t: -t[1]))
    return d


def calculate_entropy(file1, file2, file3):
    input = codecs.open(file1, mode='r', encoding='utf-8')
    output = codecs.open(file2, mode='w', encoding='utf-8')
    r1 = u'[—：（）；a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    word_tuer = ['的', '我', '你', '是', '得']
    # 去除标点符号

    for line in input.readlines():
        string = re.sub(r1, "", line)
        output.write(string)
        # print(string)

    word_lst = []
    word_dict = {}

    with open(file2) as wf:
        for word in wf.readlines():
            if word.split() and word.split() != ' +':
                word = re.sub('\n', '', word)
                word = re.sub(' +', ' ', word)
                word_lst.append(word.split(" "))

        for item in word_lst:
            for item2 in item:
                if not (item2.isspace()) and item2.strip() not in ['\u3000', '\ufeff', '']:
                    if item2 not in word_dict:
                        word_dict[item2] = 1
                    else:
                        word_dict[item2] += 1

        word_dict = sort_by_count(word_dict)  # 怒排序

    freq_sum = 0
    entropy = 0.0
    with open(file3, 'w') as wf2:
        for key in word_dict:
            if key not in word_tuer:  # 只统计杀死词列表之外的
                freq_sum += word_dict[key]
        for key in word_dict:
            if key not in word_tuer:  # 只统计杀死词列表之外的
                p = float(word_dict[key]) / float(freq_sum)
                entropy += -p * math.log(p)
            wf2.write(key + ' ' + str(word_dict[key]) + ' ' + str(float(word_dict[key])/float(freq_sum)) + '\n')

        print('Entropy of this file is ' + str(entropy))


if __name__ == '__main__':
    file_1 = 'seg12.txt'
    deal_file(file_1)
    file_1 = 'seg12.txt'
    file_2 = 'tmp.txt'
    file_3 = 'word.txt'
    calculate_entropy(file_1, file_2, file_3)

