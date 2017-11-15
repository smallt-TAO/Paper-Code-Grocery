'''
store the network for three try.

'''

from __future__ import print_function

import tensorflow as tf 
from tensorflow.contrib import rnn

# network parameters
n_input = 784 #mnist data input
n_classes = 10 # mnist total classes
dropout = 0.75 # dropout, probability to keep units


# create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
	# conv2d wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
# maxpool2d wrapper	
	return tf.nn.max_pool(x, ksize=[1, k , k, 1], strides=[ 1, k, k, 1], padding='SAME')


def network(x, n_classes, dropout=None, model="CNN"):
    if model == "CNN":
        weights = {
            'wc1' : tf.Variable(tf.random_normal([5, 5, 1, 32])),
            'wc2' : tf.Variable(tf.random_normal([5, 5, 32, 32])),
            'wc3' : tf.Variable(tf.random_normal([5, 5, 32, 64])),
            'wd1' : tf.Variable(tf.random_normal([3*584*64, 1024])),
            'out': tf.Variable(tf.random_normal([1024, n_classes]))
        }

        biases = {
            'bc1' : tf.Variable(tf.random_normal([32])),
            'bc2' : tf.Variable(tf.random_normal([32])),
            'bc3' : tf.Variable(tf.random_normal([64])),
            'bd1' : tf.Variable(tf.random_normal([1024])),
            'out' : tf.Variable(tf.random_normal([n_classes]))
        }

        x = tf.reshape(x, shape = [-1, 30, 3500, 1])

        conv1 = conv2d(x , weights['wc1'], biases['bc1'])
        conv1 = maxpool2d(conv1, k=2)
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = maxpool2d(conv2, k=2)
        conv2 = conv2d(conv2, weights['wc3'], biases['bc3'])
        conv2 = maxpool2d(conv2, k=2)

        # fully connected layer
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        # apply dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        #output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    if model == "RNN":
        n_hidden = 32
        weights = {
            'out' : tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
        }
        biases = {
            'out' : tf.Variable(tf.random_normal([n_classes]))
        }

        # prepare data shape to match bidirectional rnn function requirements
        # current data input shape: (batch_size, n_steps, n_input)
        # required shape: n_steps tensors list of shape(batch_size, n_input)
        # unstack to get a list of n_steps tensors of shape (batch_size, n_input)
        x = tf.reshape(x, shape = [-1, 30, 3500])
        x = tf.transpose(x, [0, 2, 1])
        x = tf.unstack(x, 3500, 1)

        #define lstm cells with tensorflow
        # forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
        #Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        #get lstm cell output
        try:
            output, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

        except Exeption: 
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

        #linear activation, using rnn inner loop last ouput
        return tf.matmul(output[-1], weights['out']) + biases['out']

