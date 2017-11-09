from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.contrib.slim.nets import inception

tf.app.flags.DEFINE_string('ps_hosts','','parameter servers')
tf.app.flags.DEFINE_string('worker_hosts','','workers')
tf.app.flags.DEFINE_string('job_name','','name of job')
tf.app.flags.DEFINE_integer('task_index',0,'index of task')
FLAGS = tf.app.flags.FLAGS

batch_size = 8
RMSPROP_DECAY = 0.9    
RMSPROP_MOMENTUM = 0.9        
RMSPROP_EPSILON = 1.0

ps_hosts = FLAGS.ps_hosts.split(',')
worker_hosts = FLAGS.worker_hosts.split(',')
cluster = tf.train.ClusterSpec({'ps':ps_hosts,'worker':worker_hosts})
server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)
if FLAGS.job_name == 'ps':
  server.join()
elif FLAGS.job_name == 'worker':
  with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d'%FLAGS.task_index,cluster=cluster)):
    image_batch = tf.placeholder(tf.float32,(None,224,224,3),name='images')
    label_batch = tf.placeholder(tf.int32,(None,),name='labels')
    logits,end_points = inception.inception_v3(image_batch,num_classes=100)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batch))
    ini_lr = tf.Variable(0.01,name='ini_lr',trainable=False)
    global_step = tf.Variable(0,name='global_step',trainable=False)
    learning_rate = tf.train.exponential_decay(ini_lr,global_step,1000,0.16,staircase=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate,RMSPROP_DECAY,momentum=RMSPROP_MOMENTUM,epsilon=RMSPROP_EPSILON).minimize(loss)
    prediction = tf.argmax(end_points['Predictions'],1)
    init_op = tf.global_variables_initializer()
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index==0),logdir='./aaa',init_op=init_op)
  
    with sv.prepare_or_wait_for_session(server.target) as sess:
  #    tf.global_variables_initializer().run()
#      all_images,all_labels = inception_data.get_data()
      #idx = np.random.permutation(all_images.shape[0])
      for epoch in range(100):
      #  images = all_images[idx]
      #  labels = all_labels[idx]
        for step in range(10000):
 #         result,ls,pre,_ = sess.run([logits,loss,prediction,optimizer],feed_dict={image_batch:images[step*batch_size:(step+1)*batch_size],label_batch:labels[step*batch_size:(step+1)*batch_size]})    
          result,ls,pre,_ = sess.run([logits,loss,prediction,optimizer],feed_dict={image_batch:np.random.randint(225,size=[batch_size,224,224,3]),label_batch:np.random.randint(100,size=[batch_size,])})    
          print(ls)








