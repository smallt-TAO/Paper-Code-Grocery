from __future__ import print_function
import network
import utils

import tensorflow as tf 

tf.app.flags.DEFINE_string('model', 'CNN_v1','parameter choice of model')
tf.app.flags.DEFINE_integer('n_classes', 6, 'number of classes')
FLAGS = tf.app.flags.FLAGS

# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 32
display_step = 32
dropout = 0.75 # dropout, probability to keep units

def load_dataset():
    data_all = utils.file_batch("data")
    clip = int(0.8 * len(data_all))
    train_list = data_all[:clip]
    test_list = data_all[clip:]
    return train_list, test_list

def train_model():
    height, width = 30, 3500
    x = tf.placeholder(tf.float32, [None, height, width])
    y = tf.placeholder(tf.float32, [None, FLAGS.n_classes])
    keep_prob = tf.placeholder(tf.float32) # dropout

    pred = network.network(x, FLAGS.n_classes, keep_prob, model=FLAGS.model)

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

        try:
            saver.restore(sess, 'checkpoint/' + FLAGS.model + '_model.ckpt')
            print("load checkpoint sucess")
        except:
            print("load checkpoint failed")

        step = 1
        average_cost = 0.0
        train_data, test_data = load_dataset()
        test_image, test_label = utils.read_batch_list(test_data)

        #keep training until reach max iterations
        while step < training_iters:
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            total_batch = int(len(train_data) / batch_size)
            for i in range(total_batch):
                batch_x, batch_y = utils.read_batch_list(train_data[i * batch_size:(i + 1) * batch_size])
                #run optimization op (backprop)
                sess.run(optimizer, feed_dict={x:batch_x, y:batch_y, keep_prob:dropout})

                if step % display_step == 0:
                    # calculate batch loss and accuracy
                    loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y, keep_prob:1.})
                    print("iter", str(step), " minibatch loss =" , "{:.6f}".format(loss) ," Training Accuracy =" ,"{:.5f}".format(acc))
                    saver.save(sess, 'checkpoint/' + FLAGS.model + '_model.ckpt')

            step += 1
        print("optimization finished")

        #Calculate accuracy for 256 mnist test images
        print ("testing accuracy:", sess.run(accuracy, feed_dict={x:test_image, y:test_label, keep_prob:1.}))

def main(_):
    train_model()

if __name__ == "__main__":
    tf.app.run()


