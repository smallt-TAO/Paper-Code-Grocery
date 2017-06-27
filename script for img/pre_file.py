# -*- coding: utf-8 -*-
import codecs
import os
import jieba
import re


def pre_sku(file_path):
    r1 = u'[—：（）；’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    counter_term = 0

    path_dir = os.listdir(file_path)
    for all_dir in path_dir:
        all_dir = os.path.join(file_path, all_dir)
        input_file = codecs.open(all_dir, mode='r', encoding='utf-8')
        text1seg = []
        file_result = 'sku_txt\\seg_result_{}.txt'.format(str(counter_term))
        output_file = codecs.open(file_result, mode='w', encoding='utf-8')
        for line0 in input_file.readlines():
            # line_item_sku_id = line0.strip().split('\t')
            line_item_name = line0.strip().split('\t')[1].split(' ')[0]
            line_item_name = re.sub(r1, "", line_item_name.strip())
            for line_word in jieba.cut(line_item_name):
                if line_word not in text1seg:
                    print(len(text1seg))
                    output_file.write(line_word + ' ')
                    text1seg.append(line_word)
            if len(text1seg) > 10000:
                break
        output_file.close()
        counter_term += 1


if __name__ == '__main__':
    File_path = 'sku_image'
    pre_sku(File_path)
