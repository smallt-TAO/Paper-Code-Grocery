'''
This code designed for pred the point
A Multilayer Perceptron implementation example using TensorFlow library.
'''

from __future__ import print_function
import tensorflow as tf 
import dataset

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

# gen the data set for train and test
all_x, all_y = dataset.load_train_data("train.csv")
clip_num = int(len(all_y) * 0.8)
input_x, input_y = all_x[:clip_num], all_y[:clip_num]
test_x, test_y = all_x[clip_num:], all_y[clip_num:]

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

#launch the graph
with tf.Session() as sess:
    sess.run(init)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

    if checkpoint:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint load")
    else:
        print("load checkpoint failed")

	#training cycle
    for epoch in range(training_epochs):
        current_step = epoch
        avg_cost = 0.

        total_batch = int(len(input_y)/batch_size)
		# Loop over all batches
        for i in range(total_batch):
            batch_x = input_x[i * batch_size:(i + 1) * batch_size]
            batch_y = input_y[i * batch_size:(i + 1) * batch_size]
            # run optimization op (backprop) and cost op (loss value)
            _, c = sess.run([optimizer,cost], feed_dict = {x : batch_x, y : batch_y})

			#compute average loss
            avg_cost += c/total_batch

		#display logs per epoch step
        if (epoch + 1) % display_step ==0:
            print("epoch:", '%d'% (epoch +1), "cost =", "{:.9f}".format(avg_cost)),
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy", accuracy.eval({x:test_x, y: test_y}))

        if (epoch + 1) % 50 == 0:
            print("Save the model")
            saver.save(sess, 'checkpoint/mlp_model.ckpt')

    print("optimization finished")

	#test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))

	#calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy", accuracy.eval({x:test_x, y: test_y}))
