# -*- coding: utf-8 -*-

import re
import codecs


def deal_file(file1):
    output_file = codecs.open(file1, mode='w', encoding='utf-8')

    text1seg = []
    for i in range(1):
        file1 = '.\\123.txt'
        print str(file1)
        input_file = codecs.open(file1, mode='r', encoding='utf-8')

        for line in input_file.readlines():
            r1 = u'[—：（）；a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~ ]'
            line = re.sub(r1, "", line)
            line_seg = ''.join([f + ' ' for f in line])
            # line_seg = jieba.cut(line)
            text1seg.extend(line_seg)
        input_file.close()

    for word in text1seg:
        output_file.write(word + ' ')
    output_file.close()


if __name__ == '__main__':
    file_1 = 'seg12.txt'
    deal_file(file_1)
