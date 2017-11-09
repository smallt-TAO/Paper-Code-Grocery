# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Script to train the Attention OCR model.

A simple usage example:
python train.py
"""
import collections
import logging
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags
from tensorflow.contrib.tfprof import model_analyzer
import distance
import data_provider
import common_flags
import numpy as np
import prehandle_data

FLAGS = flags.FLAGS
common_flags.define()

# yapf: disable
flags.DEFINE_integer('task', 0,
                     'The Task ID. This value is used when training with '
                     'multiple workers to identify each worker.')

flags.DEFINE_integer('ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then'
                     ' the parameters are handled locally by the worker.')

flags.DEFINE_integer('save_summaries_secs', 60,
                     'The frequency with which summaries are saved, in '
                     'seconds.')

flags.DEFINE_integer('save_interval_secs', 2,
                     'Frequency in seconds of saving the model.')

flags.DEFINE_integer('max_number_of_steps', int(1000),
                     'The maximum number of gradient steps.')

flags.DEFINE_string('checkpoint_inception', '',
                    'Checkpoint to recover inception weights from.')

flags.DEFINE_float('clip_gradient_norm', 2.0,
                   'If greater than 0 then the gradients would be clipped by '
                   'it.')

flags.DEFINE_bool('sync_replicas', False,
                  'If True will synchronize replicas during training.')

flags.DEFINE_integer('replicas_to_aggregate', 1,
                     'The number of gradients updates before updating params.')

flags.DEFINE_integer('total_num_replicas', 1,
                     'Total number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_boolean('reset_train_dir', False,
                     'If true will delete all files in the train_log_dir')

flags.DEFINE_boolean('show_graph_stats', False,
                     'Output model size stats to stderr.')
# yapf: enable

TrainingHParams = collections.namedtuple('TrainingHParams', [
    'learning_rate',
    'optimizer',
    'momentum',
    'use_augment_input',
])


def get_training_hparams():
  return TrainingHParams(
      learning_rate=FLAGS.learning_rate,
      optimizer=FLAGS.optimizer,
      momentum=FLAGS.momentum,
      use_augment_input=FLAGS.use_augment_input)


def create_optimizer(hparams):
  """Creates optimized based on the specified flags."""
  if hparams.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        hparams.learning_rate, momentum=hparams.momentum)
  elif hparams.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(hparams.learning_rate)
  elif hparams.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(hparams.learning_rate)
  elif hparams.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(hparams.learning_rate)
  elif hparams.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        hparams.learning_rate, momentum=hparams.momentum)
  return optimizer


def train(loss, init_fn, hparams):
  """Wraps slim.learning.train to run a training loop.

  Args:
    loss: a loss tensor
    init_fn: A callable to be executed after all other initialization is done.
    hparams: a model hyper parameters
  """
  optimizer = create_optimizer(hparams)

  if FLAGS.sync_replicas:
    replica_id = tf.constant(FLAGS.task, tf.int32, shape=())
    optimizer = tf.LegacySyncReplicasOptimizer(
        opt=optimizer,
        replicas_to_aggregate=FLAGS.replicas_to_aggregate,
        replica_id=replica_id,
        total_num_replicas=FLAGS.total_num_replicas)
    sync_optimizer = optimizer
    startup_delay_steps = 0
  else:
    startup_delay_steps = 0
    sync_optimizer = None

  train_op = slim.learning.create_train_op(
      loss,
      optimizer,
      summarize_gradients=True,
      clip_gradient_norm=FLAGS.clip_gradient_norm)
  return train_op

def prepare_training_dir():
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    logging.info('Create a new training directory %s', FLAGS.train_log_dir)
    tf.gfile.MakeDirs(FLAGS.train_log_dir)
  else:
    if FLAGS.reset_train_dir:
      logging.info('Reset the training directory %s', FLAGS.train_log_dir)
      tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
      tf.gfile.MakeDirs(FLAGS.train_log_dir)
    else:
      logging.info('Use already existing training directory %s',
                   FLAGS.train_log_dir)


def calculate_graph_metrics():
  param_stats = model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  return param_stats.total_parameters

def idx_to_words(idx):
  f = open('./vocab.txt')
  line = f.readline()
  f.close()
  words = list()
  if 1 not in idx and 2 not in idx:
    for num in idx:
      if num not in [0,1,2]:
        words.append(line[num-3])
  if 1 in idx and  2 not in idx:
    for num in idx[idx.index(1):]:
      if num not in [0,1,2]:
        words.append(line[num-3])
  if 1 not in idx and 2 in idx:
    for num in idx[:idx.index(2)]:
      if num not in [0,1,2]:
        words.append(line[num-3])
  if 1 in idx and 2 in idx:
    if idx.index(1) < idx.index(2):
      for num in idx[idx.index(1):idx.index(2)]:
        if num not in [0,1,2]:
          words.append(line[num-3])

  return words
  


def main(_):
  prepare_training_dir()
  num_char_classes = 77
#less than 37
  max_sequence_length = 25
  num_of_views = 4
#  null_code = 42 
  null_code = 2
  model = common_flags.create_model(num_char_classes,
                                    max_sequence_length,
                                    num_of_views,null_code)
  hparams = get_training_hparams()

  # If ps_tasks is zero, the local device is used. When using multiple
  # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
  # across the different devices.
  device_setter = tf.train.replica_device_setter(
      FLAGS.ps_tasks, merge_devices=True)
  with tf.device(device_setter):
    images_orig = tf.placeholder(tf.float32, [64, 150, 600,3], name='image_orig')
    labels = tf.placeholder(tf.int32, [64,max_sequence_length, ], name='label')
    data = data_provider.get_data(
        images_orig,labels,
        num_of_views,
        num_char_classes)
  #      augment=hparams.use_augment_input,
  #      central_crop_size=None)
    endpoints = model.create_base(data.images,data.labels_one_hot)
    total_loss = model.create_loss(labels, endpoints)
    init_fn = model.create_init_fn_to_restore(FLAGS.checkpoint,
                                              FLAGS.checkpoint_inception)
    train_op = train(total_loss, init_fn, hparams)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        images_dataset,labels_dataset = prehandle_data.get_data()
        print('data prepared')
        for epoch in range(400):
          ids = np.random.permutation(images_dataset.shape[0])
          images_permutation = images_dataset[ids]
          labels_permutation = labels_dataset[ids]
          for step in range(images_dataset.shape[0] / 64): 
            result, loss, _ = sess.run([endpoints.predicted_chars,total_loss,train_op],feed_dict={images_orig:images_permutation[step*64:(step+1)*64],labels:labels_permutation[step*64:(step+1)*64]})
           #  print(loss)
            if step % 200 == 0:
              print(loss)
              target = labels_permutation[step*64:(step+1)*64]
              leven_distance = list()
              count = 0
              for h_shape in range(result.shape[0]):
                tmp_result = list(result[h_shape])
                target_result = list(target[h_shape])
                if count < 5:
                  f = open('./aaa.txt','a+')
                  f.write(''.join(idx_to_words(tmp_result))+'\n')
                  f.write(''.join(idx_to_words(target_result))+'\n')
                  f.write('--------------------------------------'+'\n')
                  f.close()
                  count += 1
                target_start = target_result.index(1)+1
                target_end = target_result.index(2)
                divide = float(target_end - target_start)
                if 1 not in tmp_result and 2 not in tmp_result:
                  leven_distance.append((target_end-target_start)/divide)
                if 1 in tmp_result and 2 not in tmp_result:
                  leven_distance.append(distance.levenshtein(tmp_result[tmp_result.index(1)+1:],target_result[target_start:target_end])/divide)
                if 1 not in tmp_result and 2 in tmp_result:
                  leven_distance.append(distance.levenshtein(tmp_result[:tmp_result.index(2)],target_result[target_start:target_end])/divide)
                if 1 in tmp_result and 2 in tmp_result:
                  if(tmp_result.index(1) < tmp_result.index(2)):
                    leven_distance.append(distance.levenshtein(tmp_result[tmp_result.index(1)+1:tmp_result.index(2)],target_result[target_start:target_end])/divide)
                  else:
                    leven_distance.append((target_end-target_start)/divide)
              summary = sum(leven_distance) / 64
              print(summary) 
              f = open('result_1.txt','a+')
              f.write(str(loss)+'\n')
              f.write(str(summary)+'\n')
              f.close()
          if epoch % 40 == 0:
            saver.save(sess,'./trained/my-model',global_step=epoch)
          


if __name__ == '__main__':
  app.run()
