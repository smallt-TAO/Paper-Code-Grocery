# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import random
import os
import re
import codecs


class ImageChar(object):
    def __init__(self, fontcolor=(0, 0, 0), size=(100, 40),
                 fontpath='rsndm.otf', fontsize=20):
        self.size = size
        self.fontPath = fontpath
        self.fontSize = fontsize
        self.fontColor = fontcolor
        self.font = ImageFont.truetype(self.fontPath, self.fontSize)
        self.image = Image.open('381339.jpg')

    @staticmethod
    def crop_book(size):
        all_file = os.listdir(r'back_ground')
        index_book = random.randint(0, len(all_file) - 1)
        choose_file = os.path.join('back_ground', all_file[index_book])
        img = Image.open(choose_file)
        out_img = img.resize(size)
        return out_img

    def merge_image(self, image_path, pos=(480, 70),
                    pos1=(510, 80), pos2=(540, 90), image_size=(170, 220)):
        img = self.crop_book(self.size)
        merge_img = Image.open(image_path)
        merge_img = merge_img.resize(image_size)
        for merge_i in range(merge_img.size[0]):
            for merge_j in range(merge_img.size[1]):
                if merge_img.getpixel((merge_i, merge_j)) != (255, 255, 255):
                    img.putpixel((merge_i + pos[0], merge_j + pos[1]),
                                 merge_img.getpixel((merge_i, merge_j)))
                    img.putpixel((merge_i + pos1[0], merge_j + pos1[1]),
                                 merge_img.getpixel((merge_i, merge_j)))
                    img.putpixel((merge_i + pos2[0], merge_j + pos2[1]),
                                 merge_img.getpixel((merge_i, merge_j)))
        return img

    @staticmethod
    def rand_rgb():
        random_color = [(0, 0, 0), (255, 255, 255),
                        (254, 67, 101),  # 深红色
                        (249, 205, 173),  # 浅黄色
                        (131, 175, 155),  # 淡青色
                        (252, 157, 154),  # 浅红色
                        (200, 200, 169),  # 浅青色
                        (3, 38, 58), (222, 125, 44),
                        (174, 221, 129), (107, 194, 53),
                        (32, 36, 46), (6, 128, 67),
                        (90, 13, 67), (175, 18, 100)]
        return random_color[random.randint(0, len(random_color) - 1)]

    def draw_text(self, font_size, pos, txt, fill):
        draw = ImageDraw.Draw(self.image)
        the_font = ImageFont.truetype(self.fontPath, font_size)
        draw.text(pos, txt, font=the_font, fill=fill)
        del draw

    def rand_chinese(self, image_sku_id, image_sku_des):
        gap = 1
        start = 0
        char_list = []
        self.image = self.merge_image('sku_image_file\\{}.jpg'.format(image_sku_id))
        all_word_list = make_title()
        sku_des_list = [i for i in image_sku_des if i is not ' ']
        main_x = random.randint(85, 120)
        main_y = random.randint(120, 160)

        # Main color for big word.
        center_random_word = random.randint(3, 5)
        word_index_random = random.randint(0, len(all_word_list))
        title = all_word_list[word_index_random: word_index_random + center_random_word]
        center_color = self.rand_rgb()
        center_font_size = self.fontSize
        center_x = main_x
        center_y = main_y
        for i in range(0, len(title)):
            char = title[i]
            char_list.append(char)
            x = start + center_font_size * i + gap * i
            self.draw_text(center_font_size, (x + center_x, center_y), char, center_color)

        # upper region word.
        upper_random_word = random.randint(10, 14)
        title = sku_des_list[: upper_random_word]
        upper_color = self.rand_rgb()
        upper_random_pos_x = random.randint(0, 5)
        upper_random_pos_y = random.randint(0, 5)
        upper_font_size = random.randint(18, 24)
        upper_x = main_x + upper_random_pos_x
        upper_y = main_y - int(upper_font_size / 2) - upper_random_pos_y
        for i in range(0, len(title)):
            char = title[i]
            x = start + upper_font_size * i + gap * i
            self.draw_text(upper_font_size, (x + upper_x, upper_y), char, upper_color)

        # down region word.
        down_random_word = random.randint(10, 14)
        word_index_random = random.randint(0, len(all_word_list))
        title = all_word_list[word_index_random: word_index_random + down_random_word]
        down_color = self.rand_rgb()
        down_random_pos_x = random.randint(1, 7)
        down_random_pos_y = random.randint(1, 15)
        down_font_size = random.randint(18, 24)
        down_x = main_x + down_random_pos_x
        down_y = main_y + center_font_size + down_random_pos_y
        for i in range(0, len(title)):
            char = title[i]
            x = start + down_font_size * i + gap * i
            self.draw_text(down_font_size, (x + down_x, down_y), char, down_color)

        return char_list

    def save(self, path):
        self.image.save(path)


def make_title():
    word_list = []
    with codecs.open("org_file.txt", mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            r1 = u'[—：（）；a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
            string = re.sub(r1, "", line).strip()
            for i in range(len(string)):
                if string[i] in [' ']:
                    continue
                word_list.extend(string[i])
    return word_list


def pre_sku(file_path):
    term_image = []
    sku_id_list = []
    path_dir = os.listdir(file_path)
    for all_dir in path_dir:
        all_dir = os.path.join(file_path, all_dir)

        with codecs.open(all_dir, mode='r', encoding='utf-8') as input_file:
            for line0 in input_file.readlines():
                line_item_sku_id = line0.strip().split('\t')[0]
                line_item_name = line0.strip().split('\t')[1]
                if line_item_sku_id not in sku_id_list:
                    term_image.append([line_item_sku_id, line_item_name])
                sku_id_list.append(line_item_sku_id)
                if len(term_image) > 30:
                    return term_image


if __name__ == '__main__':
    ad_sku = pre_sku('sku_image')
    for [sku_id, sku_des] in ad_sku:
        ic = ImageChar(fontcolor=(100, 211, 90), size=(790, 390), fontsize=68)
        char_list0 = ic.rand_chinese(sku_id, sku_des)
        ic.save('./chinesePic/' + str(sku_id) + ".png")

