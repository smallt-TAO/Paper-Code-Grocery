import tensorflow as tf
import data_provider
import numpy as np

image_orig = tf.placeholder(tf.int32,[None,150,360,3],name='image_orig')
label = tf.placeholder(tf.int32,[None,],name='label')
return_data = data_provider.get_data(image_orig,label,4,134)
with tf.Session() as sess:
  x = np.array(np.random.randint(10,size=[2,150,360,3]),dtype=np.int32)
  y = np.array(np.random.randint(10,size=[2,]),dtype=np.int32)
  result = sess.run(return_data,feed_dict={image_orig:x,label:y})
  print(result)



