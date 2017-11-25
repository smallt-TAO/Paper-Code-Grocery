'''
This code designed for infer the point via socket.
'''

from __future__ import print_function
import tensorflow as tf 
import dataset
import os
import socket
import sys
import logging
from thread import *

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='socket_mlp.log',
                filemode='w')

# Parameters
current_iter = tf.Variable(0)
training_epochs = 150000
batch_size = 64
display_step = 5
checkpoint_path = 'checkpoint'

# Network parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_input = 51 # data input
n_classes = 2 # the number of classes


def network(x, y):
    """
    output: pred
    """
    #Create model
    def multilayer_perceptron(x, weights, biases):
        #Hidden layer with relu activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        #hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    # store layers weights and bias
    weights = {
        'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out' : tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }

    biases = {
        'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
        'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # construct model
    pred = multilayer_perceptron(x, weights, biases)

    return pred


def caculate_p(true_list, pred_list):
    neg, neg_right, pos, pos_right = 0, 0, 0, 0
    if len(pred_list) == len(true_list):
        for i in range(len(true_list)):
            if true_list[i] == 0:
                neg += 1
                if pred_list[i] == 0:
                    neg_right += 1
            else:
                pos += 1
                if pred_list[i] == 1:
                    pos_right += 1

    logging.info("pred correct: ", float(pos_right)/(pos_right + neg - neg_right))


def print_index(pred_list):
    index_list = []
    for i in range(len(pred_list)):
        if pred_list[i] == 1:
            index_list.append(str(i + 26))
    return index_list


def add_ruler(ruler_list):
    """
    ouput = [1, 1, 1, 0, 1, ... 1, 1]
    """
    res_list = [1] * len(ruler_list)
    mid_index = int(len(ruler_list[0]) / 2)
    for i in range(len(ruler_list)):
        sub_list = ruler_list[i][mid_index - 3:mid_index + 3]
        if max(sub_list) - min(sub_list) < 0.04:
            res_list[i] = 0
    return res_list


def infer():
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    pred = network(x, y)
    pred_list = tf.argmax(pred, 1)

    #initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    #launch the graph
    with tf.Session() as sess:
        sess.run(init)
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

        saver.restore(sess, checkpoint.model_checkpoint_path)
        logging.info("checkpoint load")
        
        """
        This function is for infer a file.
        input: file path
        output: result
        """
        def infer_file(sess, csv_path, pred_list):
            # pred the file
            test_list = dataset.read_real_data(csv_path)

            # test_list = dataset.read_real_data("data/daht_c001_04_15.csv")
            test_x = []
            if len(test_list) < 53:
                logging.info("Short")
            else:
                for i in range(25, len(test_list) - 25):
                    test_x.append(test_list[i - 25: i + 26])

                pred_list = sess.run(pred_list, feed_dict={x:test_x})
                ruler_list = add_ruler(test_x) 
                pred_list = [ruler_list[i] * pred_list[i] for i in range(len(pred_list))]
                logging.info("File is: {}".format(csv_path))
                logging.info("Prediction sum: ", sum(pred_list))
                logging.info("Index is: {}".format(print_index(pred_list)))
            return ','.join(print_index(pred_list))
             
        # server    
        address = ('127.0.0.1', 31501)  
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # s = socket.socket()  
        s.bind(address)  
        s.listen(5)        
        
        """
        Via thread arrive the multi-worked
        """
        def client_thread(conn, sess, pred_list):
            # infinite loop
            while True:
                data = conn.recv(512)
                logging.info("This time :", data, "on the way")
                reply = infer_file(sess, data, pred_list)
                conn.sendall(reply)
            conn.close()

        while True:
            conn, addr = s.accept()
            logging.info("Connected with " + addr[0] + ":" + str(addr[1]))
            start_new_thread(client_thread, (conn, sess, pred_list))

        s.close() 
         
if __name__ == '__main__':
    infer()

