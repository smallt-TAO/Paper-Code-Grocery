import os
import numpy as np
from PIL import Image
import codecs

def get_data():
#  filelist = os.listdir('./chinese_data/ad_flag')
#  for i in range(len(filelist)):
#    filelist[i] = filelist[i].decode('utf-8')
  filelist = os.listdir('./real_train')
#  f = codecs.open('./chinese_data/chinese_vocab.txt','r','utf-8')
  f = open('./vocab.txt')
  line = f.readline()
  f.close()
  word_to_idx = dict()
  for i in range(len(line)):
    word_to_idx[line[i]] = i+3
  images = list()
  labels = list()
  for filename in  filelist:
 #   image = Image.open(os.path.join('./chinese_data/ad_flag',filename))
    image = Image.open(os.path.join('./real_train',filename))
    images.append(np.resize(np.array(image),(150,600,3)))
    label = list()
    label.append(1)
    for ch in filename[:-4]:
      label.append(word_to_idx[ch])
    label.append(2)
    if len(label) < 25:
      for i in range(25-len(label)):
        label.append(0)
    labels.append(np.array(label))
  images = np.array(images,dtype=np.float32)
  labels = np.array(labels)
  return images,labels

def idx_to_word(idx):
  f = open('./vocab.txt')
  line = f.readline()
  print('line:%s'%line)
  f.close()
  words = list()
  for num in idx:
    words.append(line[num-3])
  return words

"""

if __name__ == '__main__':
  images,labels = get_data()
  print(labels[0])
  label = list(labels[0])
#  start = label.index(1)+1
#  end = label.index(2)
#  words = idx_to_word(label[start:end])
#  print(words)
"""
