#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recurrent network example.
"""
from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne import layers
from lasagne.nonlinearities import rectify, softmax, sigmoid, tanh

floatX = theano.config.floatX

MIN_LENGTH = 50
MAX_LENGTH = 55
N_HIDDEN = 100
N_BATCH = 100
LEARNING_RATE = .001
GRAD_CLIP = 100
EPOCH_SIZE = 100
NUM_EPOCHS = 2


def gen_data(max_length=55, n_batch=100):
    # Generate X - we'll fill the last dimension later
    X = np.random.uniform(size=(n_batch, max_length, 1), low=0.0, high=1.0).astype(np.float32)
    y = np.ones((n_batch,))

    X = X.reshape((n_batch, max_length, 1))
    y = y.reshape((n_batch,)).astype(np.int32)

    return X, y


def main(num_epochs=NUM_EPOCHS):
    input_var = T.tensor3('input_var')
    answer_var = T.ivector('answer_var')

    print("==> building network")
    example = np.random.uniform(size=(N_BATCH, MAX_LENGTH, 1), low=0.0, high=1.0).astype(np.float32)

    # InputLayer
    network = layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, 1), input_var=input_var)
    print(layers.get_output(network).eval({input_var: example}).shape)

    # GRULayer
    network = layers.GRULayer(incoming=network, num_units=50, only_return_final=True)
    print(layers.get_output(network).eval({input_var: example}).shape)

    # Last layer: classification
    network = layers.DenseLayer(incoming=network, num_units=1, nonlinearity=softmax)
    print(layers.get_output(network).eval({input_var: example}).shape)

    params = layers.get_all_params(network, trainable=True)
    prediction = layers.get_output(network, deterministic=True)
    prediction = T.argmax(prediction, axis=1)

    loss = lasagne.objectives.categorical_crossentropy(prediction, answer_var).mean()

    # difference the way to update.
    # updates = lasagne.updates.adadelta(self.loss, self.params)
    updates = lasagne.updates.momentum(loss, params, learning_rate=0.0005)

    print("==> compiling train_fn")
    train_fn = theano.function(inputs=[input_var, answer_var],
                               outputs=[prediction, loss], updates=updates)

    print("==> compiling test_fn")
    test_fn = theano.function(inputs=[input_var, answer_var], outputs=[prediction, loss])

    print("Training ...")
    try:
        for epoch in range(num_epochs):
            X, y = gen_data()
            for _ in range(EPOCH_SIZE):
                train_fn(X, y)
            pre, loss = test_fn(X, y)
            print("Epoch {} validation pred = {}".format(epoch, pre))
            print("         validation cost = {}".format(loss))
    except KeyboardInterrupt:
        pass

    # print("Predicted ...")
    # try:
    #     t_test, y_test = gen_data()
    #     print(t_test)
    #     print(y_test)
    #     predict = predicted_test(t_test, y_test)
    #     print(predict)
    # except KeyboardInterrupt:
    #     pass

if __name__ == '__main__':
    main()
