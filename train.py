import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np

#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


batch_size = 32

#Prepare input data
classes = ['squares', 'circles', 'triangles']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
train_path='./training_data_v2'

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))



session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)



##Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
fc_layer_size = img_size

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters,
               layer_name):  
    
    with tf.name_scope(layer_name):
      ## We shall define the weights that will be trained using create_weights function.
      with tf.name_scope('weights'):
        weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        variable_summaries(weights)
      ## We create biases using the create_biases function. These are also trained.
      with tf.name_scope('biases'):
        biases = create_biases(num_filters)
        variable_summaries(biases)


      ## Creating the convolutional layer
      layer = tf.nn.conv2d(input=input,
                      filter=weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')

      layer += biases

      ## We shall be using max-pooling.  
      layer = tf.nn.max_pool(value=layer,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
      ## Output of pooling is fed to Relu which is the activation function for us.
      layer = tf.nn.relu(layer, name='activation')
      tf.summary.histogram('activation', layer)

      return layer
    

def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    with tf.name_scope('flatten_layer'):
      layer_shape = layer.get_shape()

      ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
      num_features = layer_shape[1:4].num_elements()

      ## Now, we Flatten the layer so we shall have to reshape to num_features
      layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             layer_name,
             use_relu=True):
    with tf.name_scope(layer_name):
      #Let's define trainable weights and biases.
      with tf.name_scope('weights'):
        weights = create_weights(shape=[num_inputs, num_outputs])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = create_biases(num_outputs)
        variable_summaries(biases)

      # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
      layer = tf.matmul(input, weights) + biases
      if use_relu:
          layer = tf.nn.relu(layer, name='activation')

      tf.summary.histogram('activation', layer)

    return layer


layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1,
               layer_name='conv1')
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2,
               layer_name='conv2')

# layer_conv3= create_convolutional_layer(input=layer_conv2,
#                num_input_channels=num_filters_conv2,
#                conv_filter_size=filter_size_conv3,
#                num_filters=num_filters_conv3,
#                layer_name='conv3')
          
layer_flat = create_flatten_layer(layer_conv2)

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True,
                     layer_name='fc1')

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False,
                     layer_name='fc2') 

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
with tf.name_scope('cross_entropy'):
  with tf.name_scope('total'):
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    # labels=y_true)
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_true_cls, logits=layer_fc2)
tf.summary.scalar('cross_entropy', cross_entropy)
# cost = tf.reduce_mean(cross_entropy)
with tf.name_scope('train'):
  optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    with tf.name_scope('train'):
      acc = session.run(accuracy, feed_dict=feed_dict_train)
      val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
      tf.summary.scalar('accuracy', val_acc)
      msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
      print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(layer_fc2, 1), y_true_cls)
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to
# /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./logs/train', session.graph)
test_writer = tf.summary.FileWriter('./logs/test')

saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        if i % int(data.train.num_examples/batch_size) == 0: 
          summary, val_loss = session.run([merged, cross_entropy], feed_dict=feed_dict_val)
          epoch = int(i / int(data.train.num_examples/batch_size))    
          
          show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
          saver.save(session, './shapes-model') 
          test_writer.add_summary(summary, i)
        else:
          if i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = session.run([merged, optimizer], feed_dict=feed_dict_tr, options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
          else:
            summary, _ = session.run([merged, optimizer], feed_dict=feed_dict_tr)
            train_writer.add_summary(summary, i)

    total_iterations += num_iteration

train(num_iteration=300)
train_writer.close()
test_writer.close()