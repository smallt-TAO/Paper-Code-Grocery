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


MIN_LENGTH = 50
MAX_LENGTH = 55
N_HIDDEN = 100
N_BATCH = 100
LEARNING_RATE = .001
GRAD_CLIP = 100
EPOCH_SIZE = 100
NUM_EPOCHS = 2


def gen_data(max_length=MAX_LENGTH, n_batch=N_BATCH):
    # Generate X - we'll fill the last dimension later
    X = np.array([[[1, np.random.randint(5)] for i in range(max_length)] for j in range(n_batch)])
    y = np.array([0 for i in range(n_batch)])

    X = X.reshape((n_batch, max_length, 2))
    y = y.reshape((n_batch,))

    return X, y


def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")

    target_values = T.ivector('target_output')

    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, 2))

    l_concat = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)

    l_out = lasagne.layers.DenseLayer(l_concat, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)

    test_prediction = lasagne.layers.get_output(l_out, deterministic=True)

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_values), dtype=theano.config.floatX)
    predicted_values = test_prediction.flatten()
    cost = lasagne.objectives.categorical_crossentropy(predicted_values, target_values).mean()
    all_params = lasagne.layers.get_all_params(l_out)

    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost, updates=updates)
    compute_cost = theano.function([l_in.input_var, target_values], cost)
    predicted_test = theano.function([l_in.input_var, target_values], test_acc, on_unused_input='warn')

    X_val, y_val = gen_data()

    print("Training ...")
    try:
        for epoch in range(num_epochs):
            for _ in range(EPOCH_SIZE):
                X, y = gen_data()
                train(X, y)
            cost_val = compute_cost(X_val, y_val)

            print("Epoch {} validation cost = {}".format(epoch, cost_val))
    except KeyboardInterrupt:
        pass

    print("Predicted ...")
    try:
        t_test, y_test = gen_data()
        print(t_test)
        print(y_test)
        predict = predicted_test(t_test, y_test)
        print(predict)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
