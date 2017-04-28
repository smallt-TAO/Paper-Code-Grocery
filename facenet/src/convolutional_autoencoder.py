"""Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import math
import facenet
import lfw
import argparse
from tensorflow.python.ops import data_flow_ops
import cv2
import logging
# from libs.activations import lrelu
# from libs.utils import corrupt

def corrupt(x):
    """Take an input tensor and add uniform masking.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.

    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))

def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.

    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.

    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def autoencoder(input_shape=[None, 160,160,3],
                n_filters=[1, 10, 10, 10],
                filter_sizes=[3, 3, 3, 3],
                corruption=False):
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training

    Raises
    ------
    ValueError
        Description
    """

    # input to the network
    x = tf.placeholder(tf.float32, input_shape, name='x')

    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # Optionally apply denoising autoencoder
    if corruption:
        current_input = corrupt(current_input)

    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x_tensor))

    return {'x': x, 'z': z, 'y': y, 'cost': cost}

def get_dataset(paths):
    image_paths = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        # classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths += [os.path.join(facedir,img) for img in images]
    return image_paths

def parameters_counter():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        name = variable.name
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
            # print(variable_parametes)
        total_parameters += variable_parametes
        tf.logging.info("%s-----%s------%s", name, shape, variable_parametes)
    tf.logging.info('total parametes: %d' % total_parameters)
    return total_parameters

def main(args):
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    if not os.path.isdir(os.path.join(model_dir, 'images')):
        os.makedirs(os.path.join(model_dir, 'images'))

    train_set = facenet.get_dataset(args.data_dir)
    logging.info('Model directory: %s' % model_dir)
    logging.info('Log directory: %s' % log_dir)
    logging.info('Train set size:%d' % len(train_set))

    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        logging.info('Pre-trained model: %s' % pretrained_model)

    n_filters = [1]
    filter_sizes = []
    if args.n_filters:
        n_filters += [int(s) for s in args.n_filters.strip().split(',')]
        logging.info('%s' % str(n_filters))
    if args.filter_sizes:
        filter_sizes += [int(s) for s in args.filter_sizes.strip().split(',')]
        logging.info('%s' % str(filter_sizes))


    # Build graph
    logging.info('building graph...')
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        ae = autoencoder(n_filters=n_filters, filter_sizes=filter_sizes)
        optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(ae['cost'])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        parameters_counter()
        datasets = get_dataset(args.data_dir)

        batch_size = args.batch_size
        n_epochs = args.epoch_size

        batch_xs = []
        batch_num = 0

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=100)
        if pretrained_model:
            logging.info('Restoring pretrained model: %s' % pretrained_model)
            saver.restore(sess, pretrained_model)

        logging.info('training ...')
        logging.info('dataset len:%d' % len(datasets))
        for epoch_i  in range(n_epochs):
            for idx in range(len(datasets)):
                filename = datasets[idx]
                file_contents = tf.read_file(filename)
                image = tf.image.decode_jpeg(file_contents)
                # image = tf.subtract(image, 0.5)
                # image = tf.multiply(image, 2.0)
                image = tf.cast(image, tf.float32)
                batch_xs.append(image)
                if (idx+1) % batch_size == 0:
                    _batch_xs = sess.run(batch_xs)
                    train = np.array([img for img in _batch_xs])
                    if batch_num % 100 == 0:
                        logging.info('sess run x')
                        _ax,_  = sess.run([ae, optimizer], feed_dict={ae['x']: train})
                        x_ = _ax['x']
                        z_ = _ax['z']
                        y_ = _ax['y']

                        # logging.info('type x_ : %s' % type(z_))
                        # logging.info('%s, %s, %s' % (x_.shape, z_.shape, y_.shape))
                        cost_ = _ax['cost']
                        logging.info('epoch:%d, loss:%s' % (epoch_i, str(cost_)))

                        cv2.imwrite(os.path.join(model_dir, 'images/') + str(batch_num) + '_x.jpg', x_[1])
                        cv2.imwrite(os.path.join(model_dir, 'images/') + str(batch_num) + '_y.jpg', y_[1])
                    else:
                        sess.run(optimizer, feed_dict={ae['x']: train})
                    
                    batch_num += 1
                    batch_xs[:] = []
                    logging.info('batch num:%d' % batch_num)
                
                # if batch_num % 100 == 0:
                #     pass
            
            # save model
            logging.info('Saving variables')
            start_time = time.time()
            checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % subdir)
            saver.save(sess, checkpoint_path, global_step=batch_num, write_meta_graph=False)
            save_time_variables = time.time() - start_time
            logging.info('Variables saved in %.2f seconds' % save_time_variables)
            metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % subdir)
            save_time_metagraph = 0
            if not os.path.exists(metagraph_filename):
                logging.info('Saving metagraph')
                start_time = time.time()
                saver.export_meta_graph(metagraph_filename)
                save_time_metagraph = time.time() - start_time
                logging.info('Metagraph saved in %.2f seconds' % save_time_metagraph)



def test_mnist():
    """Test the convolutional autoencder using MNIST."""
    # %%
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    # %%
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder()

    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Fit all training data
    batch_size = 100
    n_epochs = 10
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # Plot example reconstructions
    n_examples = 10
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (784,)) + mean_img,
                (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_filters', type=str,
        help='layers.')
    parser.add_argument('--filter_sizes', type=str,
        help='size of each layer filter.')
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--logs_base_dir', type=str,
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='~/datasets/facescrub/fs_aligned:~/datasets/casia/casia-webface-aligned')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=96)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=float,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
