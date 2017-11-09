"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
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

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

import logging
# logging.basicConfig(level=logging.DEBUG,
#                 format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                 datefmt='%a, %d %b %Y %H:%M:%S',
#                 filename='t.log',
#                 filemode='w')
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# console.setFormatter(formatter)
# logging.getLogger('').addHandler(console)


def main(args):
  
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement = True)
        with tf.Session(config = config) as sess:
            # Get the paths for the corresponding images
            paths, actual_issame,ids = lfw.get_paths_v2(args.lfw_dir)
            logging.info('len of paths:%d' % len(paths))
            logging.info('paths[0]:%s' % paths[0])
            # Load the model
            # logging.info('Model directory: %s' % args.model_dir)
            # meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            meta_file = args.meta_file
            ckpt_file = args.ckpt_file
            logging.info('Metagraph file: %s' % meta_file)
            logging.info('Checkpoint file: %s' % ckpt_file)
            facenet.load_model_v2(meta_file, ckpt_file)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            #phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
            
            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]
        
            # Run forward pass to calculate embeddings
            logging.info('Runnning forward pass on LFW images')
            batch_size = args.lfw_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            logging.info('batch size:%d, nrof_images len:%d, nrof_batches len:%d' %(batch_size, nrof_images, nrof_batches))
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder: False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            logging.info('forward process complete.')
            logging.info('emb_array len:%d'% len(emb_array))
            best_threshold, tpr, fpr, acc, precision, recall, result = lfw.evaluate_v2(emb_array, actual_issame, ids)
            logging.warning('model:%s,%s' % (meta_file, ckpt_file))
            logging.warning('Best result: %0.3f,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f' % (best_threshold, tpr, fpr, acc, precision, recall))

            # total_parameters = 0
            # for variable in tf.trainable_variables():
            #     # shape is an array of tf.Dimension
            #     shape = variable.get_shape()
            #     name = variable.name
            #     variable_parametes = 1
            #     for dim in shape:
            #         # print(dim)
            #         variable_parametes *= dim.value
            #         # print(variable_parametes)
            #     total_parameters += variable_parametes
            #     # tf.logging.info("%s-----%s------%s", name, shape, variable_parametes)
            # tf.logging.info('total parametes: %d' % total_parameters)

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--labels_file', type=str, help='labels file path')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--ckpt_file', type=str,
        help='ckpt file')
    parser.add_argument('--meta_file', type=str,
        help='meta file')
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
