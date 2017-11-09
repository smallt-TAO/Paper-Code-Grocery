"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet
import logging

def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

def evaluate_v2(embeddings, actual_issame, ids):
    '''
    find best threshold,
    calculate tpr, fpr ,accuracy, precision, recall
    :param embeddings:
    :param actual_issame:
    :return:
    '''
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    # print('@@@@@@@@@ len of thresholds:%d' % len(thresholds))
    return facenet.calculate_roc_v2(thresholds, embeddings1, embeddings2, np.asanyarray(actual_issame), ids)

def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list

def get_paths(image_dir, val_label_file):
    '''
    get image pairs and the label
    :param image_dir:
    :param val_label_file:
    :return:
    '''
    labels_dict = {}
    with open(val_label_file, 'r') as lines:
        for line in lines:
          arr = line.strip().split('\t')
          labels_dict[arr[0].strip()] = arr[1].strip()

    path_list = []
    issame_list = []
    ids = []
    for dirpath, _, _ in os.walk(image_dir):
        files = os.listdir(dirpath)
        if len(files) == 2 and not os.path.isdir(files[0]):
            sku = files[0][:files[0].find('_')]
            if sku not in labels_dict:
                continue
            tfiles = [os.path.join(dirpath, filename) for filename in files]
            path_list += tfiles
            issame = False
            if labels_dict[sku] == 'YES':
                issame = True
            issame_list.append(issame)
            if sku not in ids:
                ids.append(sku)
    return path_list, issame_list, ids

def get_paths_v2(image_dir):
    '''
    get image pairs
    :param image_dir:
    :return:
    '''
    path_list = []
    issame_list = []
    ids = []
    for dirpath, _, _ in os.walk(image_dir):
        files = os.listdir(dirpath)
        for i, f1 in enumerate(files):
            logging.info(type(i))
            if (i+1) < len(files):
                #for j, f2 in enumerate(files[int(i) +1:]):
                for j, f2 in enumerate(files):
                    path_list.append(os.path.join(dirpath, f1))
                    path_list.append(os.path.join(dirpath, f2))
                    logging.info('%d,%d' % (i, j))
                    logging.info('%s,%s' % (f1, f2))
                    issame = False
                    issame_list.append(issame)
                    ids.append(f1[:f1.find('.')] + '_' + f2[:f2.find('.')])
    return path_list, issame_list, ids


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)



