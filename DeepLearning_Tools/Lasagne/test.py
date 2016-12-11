#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
"""

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne


# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 55
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Number of training sequences in each batch
N_BATCH = 100
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 10


def gen_data(max_length=MAX_LENGTH, n_batch=N_BATCH):
    """
    min_length : int
        Minimum sequence length.
    max_length : int
        Maximum sequence length.
    n_batch : int
        Number of samples in the batch.
    """
    # Generate X - we'll fill the last dimension later
    X = np.array([[[j, j] for i in range(max_length)]for j in np.random.randint(n_batch)])
    y = np.array([1 for i in range(n_batch)])

    X = X.reshape((n_batch, max_length, 2))
    y = y.reshape((n_batch,))

    return X, y


def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")

    target_values = T.vector('target_output')

    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, 2))

    l_concat = lasagne.layers.RecurrentLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        W_in_to_hid=lasagne.init.HeUniform(),
        W_hid_to_hid=lasagne.init.HeUniform(),
        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)

    l_out = lasagne.layers.DenseLayer(l_concat, num_units=1, nonlinearity=lasagne.nonlinearities.softmax)

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)
    predicted_values = network_output.flatten()

    # Our cost will be mean-squared error
    cost = T.mean(0.5*(predicted_values - target_values)**2)

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)

    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost, updates=updates)
    compute_cost = theano.function([l_in.input_var, target_values], cost)
    predicted_test = theano.function(
         [l_in.input_var, target_values], predicted_values, on_unused_input='warn')

    # We'll use this "validation set" to periodically check progress
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
