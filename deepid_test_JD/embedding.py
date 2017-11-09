# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:45:49 2017

@author: wulixin
"""
import pickle
from deepid1 import *
import tensorflow as tf
from scipy.spatial.distance import cosine, euclidean
from vec import load_data
import numpy as np
import os
from PIL import Image


def vectorize_imgs(img_path):
    with Image.open(img_path) as img:
        img = img.resize((80, 80))
        arr_img = np.asarray(img, dtype='float32')
        return arr_img

if __name__ == '__main__':
    test_file = "./val-160-subdir"
    X1 = []
    X2 = []
    embs_dict = []
    for sku in os.listdir(test_file):
        pic = os.listdir(test_file+'/'+sku)
        embs_dict.append(sku)
        X1.append(vectorize_imgs(test_file+'/'+ sku + '/' + pic[0]))
        X2.append(vectorize_imgs(test_file+'/'+ sku + '/' + pic[1]))
    testX1 = np.asarray(X1, dtype='float32')
    testX2 = np.asarray(X2, dtype='float32')
    all_vars = tf.trainable_variables()
    var_to_restore = [v for v in all_vars if not v.name.startswith('loss')]
    saver = tf.train.Saver(var_to_restore)    
    
    with tf.Session() as sess:
        saver.restore(sess, 'checkpoint/true_100000.ckpt')
        h1 = sess.run(h5, {h0: testX1})
        h2 = sess.run(h5, {h0: testX2})
        sim = 1 - np.array([cosine(x, y) for x, y in zip(h1, h2)])
    
    contrast = "./lfw-cto-val-t.txt"
    contrast_dict = {}
    with open(contrast, 'r') as lines:
        for line in lines:
            arr = line.strip().split('\t')
            contrast_dict[arr[0].strip()] = arr[1].strip()
    
    outfile = open('./distance.txt', 'w')
    for i,j in enumerate(embs_dict):
        label = contrast_dict[j]
        outfile.write('%s\t%s\t%s\n' % (j, label, str(sim[i])))
    outfile.close() 
            
            
