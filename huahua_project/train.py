'''
A Convolutional Network implementation example using TensorFlow library.
'''

from __future__ import print_function
import network
import utils

import tensorflow as tf 

# Parameters
learning_rate = 0.001
training_iters = 1000000
batch_size = 1
display_step = 64
checkpoint_path = 'checkpoint'

# network parameters
n_input = 784 # data input
n_classes = 6 # total classes
dropout = 0.75 # dropout, probability to keep units

def load_dataset():
    image, label = utils.fake_data(batch_size)
    return image, label

def train_model():
    x = tf.placeholder(tf.float32, [None, 30, 3500, 1])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) # dropout

    # pred = network.network(x, n_classes, keep_prob)
    pred = network.network(x, n_classes, model="RNN")

    # define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # evaluate model
    correct_pre = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

    # initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # launch the graph
    with tf.Session() as sess:
        sess.run(init)
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("load checkpoint sucess")
        else:
            print("load checkpoint failed")

        step = 1
        #keep training until reach max iterations
        while step * batch_size < training_iters:
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x, batch_y = load_dataset()
            #run optimization op (backprop)
            sess.run(optimizer, feed_dict={x:batch_x, y:batch_y, keep_prob:dropout})

            if step % display_step == 0:
                # calculate batch loss and accuracy
                loss, acc = sess.run([cost,accuracy], feed_dict={x:batch_x, y:batch_y, keep_prob:1.})

                print("iter", str(step*batch_size), " minibatch loss =" , "{:.6f}".format(loss) ," Training Accuracy =" ,"{:.5f}".format(acc))

            step += 1

        print("optimization finished")

        #Calculate accuracy for 256 mnist test images
        print ("testing accuracy:", sess.run(accuracy, feed_dict={x:mnist.test.images[:256], y:mnist.test.labels[:256], keep_prob:1.}))


if __name__ == "__main__":
    train_model()

