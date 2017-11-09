# -*- coding: utf-8 -*-

import codecs
import random


def read_file(file1):
    input_file = codecs.open(file1, mode='r', encoding='utf-8')
    word_list = []

    for line in input_file.readlines():
        line = line.strip().split(' ')
        for line_word in line:
            if line_word not in word_list and line_word is not '\ufeff':
                word_list.append(line_word)
            else:
                pass
    return word_list[1:]


def gen_text(all_list, text_size=2):
    pre_list = ""
    for i in range(text_size):
        pre_list += all_list[random.randint(0, len(all_list) - 1)]
    return pre_list[:text_size]


if __name__ == '__main__':
    file_1 = 'seg_result.txt'
    All_list = read_file(file_1)
    text = gen_text(All_list, 10)
    print(text)
