'''
This code designed for infer the point
A Multilayer Perceptron implementation example using TensorFlow library.
'''

from __future__ import print_function
import tensorflow as tf 
import dataset
import os

#Parameters
current_iter = tf.Variable(0)
training_epochs = 150000
batch_size = 64
display_step = 5
checkpoint_path = 'checkpoint'

#Network parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_input = 51 # data input
n_classes = 2 # the number of classes

#tf graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

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
# define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
learning_rate = tf.train.exponential_decay(0.03, current_iter, decay_steps=training_epochs, decay_rate=0.03)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step=current_iter)

#initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()


def caculate_p(true_list, pred_list):
    neg, neg_right, pos, pos_right = 0, 0, 0, 0
    if len(pred_list) == len(ture_list):
        for i in range(len(ture_list)):
            if ture_list[i] == 0:
                neg += 1
                if pred_list[i] == 0:
                    neg_right += 1
            else:
                pos += 1
                if pred_list[i] == 1:
                    pos_right += 1

    print("pred correct: ", float(pos_right)/(pos_right + neg - neg_right))


#launch the graph
with tf.Session() as sess:
    sess.run(init)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

    if checkpoint:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint load")
    else:
        print("load checkpoint failed")

	# test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

	# calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    test_x, test_y = dataset.load_train_data("test.csv")
    print("Accuracy", accuracy.eval({x:test_x, y: test_y}))
    pred_list = tf.argmax(pred, 1).eval({x:test_x})
    ture_list = tf.argmax(y, 1).eval({y:test_y})

    caculate_p(ture_list, pred_list)         
    
    # pred the file
    data_dir = "data"
    path_dir = os.listdir(data_dir)
    for all_file in path_dir:
        csv_path = os.path.join('%s/%s' % (data_dir, all_file))
        print(csv_path)
        test_list = dataset.read_real_data(csv_path)

        # test_list = dataset.read_real_data("data/daht_c001_04_15.csv")
        test_x = []
        if len(test_list) < 53:
            print("Short")
        else:
            for i in range(25, len(test_list) - 25):
                test_x.append(test_list[i - 25: i + 26])

            pred_list = tf.argmax(pred, 1).eval({x:test_x})
            print("File is: {}".format(csv_path))
            print("Prediction", pred_list)
            print(sum(pred_list))

            for i in range(len(pred_list)):
                if pred_list[i] == 1:
                    print(i + 26, end=' ')
            print()

