# encoding=utf-8
import os
from itertools import izip
import sys
from glob import glob
import matplotlib.image as mpimg
import re
import codecs
from xpinyin import Pinyin


def make_dict(title, word_dict, all_word):
    for wo in title.split('.')[0]:
        if wo in word_dict:
            continue
        else:
            all_word.append(wo.encode('utf-8'))
            word_dict[wo] = len(word_dict)
    return word_dict, all_word


def each_file(file_path):
    output_path = 'annotation_train_words.txt'
    word_path = 'vocab.txt'
    dict_se0 = {}
    new_lines = []
    all_word = []
    path_dir = os.listdir(file_path)

    for allDir in path_dir:
        dict_se0, all_word = make_dict(allDir.decode('utf-8'), dict_se0, all_word)
        labels = allDir.decode('utf-8').encode('utf-8').split('.')[0]
        paths = os.path.join("./", allDir.decode('utf-8').encode('utf-8'))
        new_lines.append(paths + ' ' + labels + '\n')

    with open(output_path, 'w') as out_f:
        out_f.writelines(new_lines)
    with open(word_path, 'w') as out_f:
        out_f.writelines(all_word)

    se0_dict = dict(izip(dict_se0.itervalues(), dict_se0.iterkeys()))
    return dict_se0, se0_dict, all_word


def each_file_pinyin(file_path):
    output_path = 'annotation_pinyin.txt'
    dict_se0 = {}
    new_lines = []
    all_word = []
    path_dir = os.listdir(file_path)
    P = Pinyin()

    for allDir in path_dir:
        pinyin_str = ""
        dict_se0, all_word = make_dict(allDir.decode('utf-8'), dict_se0, all_word)
        labels = allDir.decode('utf-8').split('.')[0]
        for i in range(len(labels)):
            # print(P.get_pinyin(labels[i]))
            try:
                pinyin_str += str(P.get_pinyin(labels[i]))
            except:
                pass
        paths = os.path.join("./", allDir.decode('utf-8').encode('utf-8'))
        new_lines.append(paths + ' ' + pinyin_str + '\n')

    with open(output_path, 'w') as out_f:
        out_f.writelines(new_lines)


def make_title():
    word_dict = {}
    with codecs.open("./dataset/vocab.txt", mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            r1 = u'[—：（）；a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
            string = re.sub(r1, "", line).strip()
            for i in range(len(string)):
                if string[i] in [' ']:
                    continue
                else:
                    if string[i] not in word_dict:
                        word_dict[string[i]] = len(word_dict)
    return word_dict


if __name__ == "__main__":
    File_Path = "./chinesePic"
    # a, _, _ = each_file(File_Path)
    each_file_pinyin(File_Path)
    # a = make_title()
    # print(len(a))
