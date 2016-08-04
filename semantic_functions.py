"""Functions to build the semanticimage modelling 

Following the paper:

Effective Semantic Pixel labelling with Convolutional Networks and Conditional Random Fields
Sakrapee Paisitkriangkrai1, Jamie Sherrah2, Pranam Janney2 and Anton Van-Den Hengel1
1Australian Centre for Visual Technology (ACVT), 
The University of Adelaide, 
Australia 2Defence Science and Technology Organisation (DSTO), Australia

Julia Baldauf
"""
 
from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
import pickle

def load_data(pickle_file, cwd):
	"""Loads the data that was produced in semantic_createdata from directory cwd 
	
	Args:
		pickle_file: name of file
		cwd: optional, directory where data lives
	Returns:
		images: nparray of images (pixel array)
		labels: nparray with corresponding labels
	"""
	os.chdir(cwd)
	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		images = save['dataset']
		labels = save['labels']
		del save  # hint to help gc free up memory
		print('All images and labels', images.shape, labels.shape)
	return images, labels

def data_prep(images, labels, v_size = 0.1, te_size = 0.1):
	"""Reformat into a shape that's more adapted to the models we're going to train:
		- data as a flat matrix,
		- labels as float 1-hot encodings.
	Args:
		images: nparray
		labels: nparray
	Returns:
		images: nparray of images (pixel array)
		labels: nparray with corresponding 1-hot encodings labels
	"""
	t_size = 1-v_size-te_size
        image_size = images.shape[1]
	dim = images.shape[3]
	num_labels = max(labels)+1
	images = images.reshape((-1, image_size,image_size,dim)).astype(np.float32)
	# Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	training_size = int(images.shape[0] * t_size)
	validation_size = int(images.shape[0] * v_size)
	test_size = int(images.shape[0] * te_size)
	train_images = images[0:training_size]
	train_labels = labels[0:training_size]
	valid_images = images[training_size+1:training_size+validation_size]
	valid_labels = labels[training_size+1:training_size+validation_size]
	test_images = images[training_size+validation_size+1:training_size+validation_size+test_size]
	test_labels = labels[training_size+validation_size+1:training_size+validation_size+test_size]
	print('Training set', train_images.shape, train_labels.shape)
	print('Validation set', valid_images.shape, valid_labels.shape)
	print('Test set', test_images.shape, test_labels.shape)
	return train_images, train_labels,valid_images,valid_labels,test_images,test_labels

def weight_variable(shape,stddev=0.1):
	'''
	One should generally initialize weights with a small amount of noise for symmetry breaking, 
	and to prevent 0 gradients. Since we're using ReLU neurons, 
	it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons." 
	'''
	initial = tf.truncated_normal(shape, stddev)
	return tf.Variable(initial)

def variable_with_weight_decay(shape, stddev, wd):
	"""Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.
	tf.nn.l2_loss() Computes half the L2 norm of a tensor without the sqrt: output = sum(t ** 2) / 2
	tf.nn.mul(x,y) Returns x * y element-wise

	Args:
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
		decay is not added for this Variable.

	Returns:
		Variable Tensor
	"""
	var = weight_variable(shape,stddev)

	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('l2_losses', weight_decay)
	return var

def bias_variable(shape):
	'''
	Initialises a bias vector 
	
	Args:
		shape: length of bias vector
	
	Returns:
		Variable Tensor
	'''
	initial = tf.constant(1.0, shape=shape)
	return tf.Variable(initial)


def conv2d(x, W):
	'''
	Convolutions uses a stride of one and are zero padded so that the output is the same size as the input. 
	
	Args:
		x: list of ints
		W: standard deviation of a truncated Gaussian
	
	Returns:
		Variable Tensor
	'''
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	'''
	Our pooling is plain old max pooling over 2x2 blocks.

	Args:
		x: list of ints
	
	Returns:
		Variable Tensor
	'''
	return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
						strides=[1, 2, 2, 1], padding='SAME')

def inference(images,fc_units =128, num_labels=5):
	"""Chooses the rigth model accorind to the input data
	Args:
		Images
		fc_units: integer, optional, number of neurons in fully connected layer
		num_labels: integer, optional, number of final number of categorizations

	Returns:
		calculated labels: logits calculated with the right model
	"""
	if images.get_shape()[1].value == 64:
		return inference64(images,fc_units, num_labels)
	if images.get_shape()[1].value == 32:
		return inference32(images,fc_units, num_labels)
	if images.get_shape()[1].value == 16:
		return inference16(images,fc_units, num_labels)

def inference64(images,fc_units, num_labels):
	"""Build the convolutional net model with 4conv and 2fc layers and drop out layers for the fc layers
	 Args:
		Images
		fc_units: integer, optional, number of neurons in fully connected layer
		num_labels: integer, optional, number of final number of categorizations

	  Returns:
		calculated labels: logits
	"""
	stddevi = 0.1
	wdecay = 0.0005
	# Define the hidden layers of the neural network
	with tf.name_scope('conv1'):
		#W_conv1 = variable_with_weight_decay([5, 5, 3, 32], stddevi, wdecay)
		W_conv1 = weight_variable([5, 5, 3, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
		h_norm1 = tf.nn.lrn(h_conv1)
		h_pool1 = max_pool_2x2(h_norm1)
		#tf.histogram_summary('wei',W_conv1)
		#tf.histogram_summary('bia',b_conv1)
  
	with tf.name_scope('conv2'):
		#W_conv2 = variable_with_weight_decay([5, 5, 32, 64],stddevi, wdecay)
		W_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		h_norm2 = tf.nn.lrn(h_conv2)
		h_pool2 = max_pool_2x2(h_norm2)
		#h_pool2 = max_pool_2x2(h_conv2)
		#_ = tf.histogram_summary('conv2_weights', h_pool2)
		#_ = tf.histogram_summary('conv2_biases', b_conv2)
	
	with tf.name_scope('conv3'):
		#W_conv3 = variable_with_weight_decay([5, 5, 64, 96],stddevi, wdecay)
		W_conv3 = weight_variable([5, 5, 64, 96])
		b_conv3 = bias_variable([96])
		h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
		h_norm3 = tf.nn.lrn(h_conv3)
		h_pool3 = max_pool_2x2(h_norm3)
		#h_pool3 = max_pool_2x2(h_conv3)
		#_ = tf.histogram_summary('conv3_weights', W_conv3)
		#_ = tf.histogram_summary('conv3_biases', b_conv3) 
	
	with tf.name_scope('conv4'):
		#W_conv4 = variable_with_weight_decay([5, 5, 96, 128],stddevi, wdecay)
		W_conv4 = weight_variable([5, 5, 96, 128])
		b_conv4 = bias_variable([128])
		h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
		h_norm4 = tf.nn.lrn(h_conv4)
		h_pool4 = max_pool_2x2(h_norm4)
		#h_pool4 = max_pool_2x2(h_conv4)
		#_ = tf.histogram_summary('conv4_weights', W_conv4)
		#_ = tf.histogram_summary('conv4_biases', b_conv4) 
	
	with tf.name_scope('fc5') as scope:
		#h_pool5_flat = tf.reshape(h_pool4, [batch_size, -1])
		shape = h_pool4.get_shape().as_list()
		h_pool5_flat = tf.reshape(h_pool4, [shape[0], shape[1] * shape[2] * shape[3]])
		w5_dim = h_pool5_flat.get_shape()[1].value
		W_fc5 = weight_variable([w5_dim, 128])
		#W_fc5 = variable_with_weight_decay([w5_dim, 128],stddevi, wdecay)
		b_fc5 = bias_variable([fc_units])
		h_fc5 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc5) + b_fc5)
		#_ = tf.histogram_summary('fc5_weights', W_fc5)
		#_ = tf.histogram_summary('fc5_biases', b_fc5)
		#dropoutlayer to reduce overfitting
		#With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, 
		#otherwise outputs 0. The scaling is so that the expected sum is unchanged.
		keep_prob_fc5 = tf.placeholder(tf.float32)
		h_fc5_drop = tf.nn.dropout(h_fc5, keep_prob_fc5)
	
	with tf.name_scope('fc6') as scope:
		W_fc6 = weight_variable([fc_units,fc_units])
		#W_fc6 = variable_with_weight_decay([fc_units,fc_units],stddevi, wdecay)
		b_fc6 = bias_variable([fc_units])
		h_fc6 = tf.matmul(h_fc5, W_fc6) + b_fc6
		#_ = tf.histogram_summary('fc6_weights', W_fc6)
		#_ = tf.histogram_summary('fc6_biases', b_fc6) 
		#dropoutlayer to reduce overfitting
		keep_prob_fc6 = tf.placeholder(tf.float32)
		h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob_fc6)
  
	#To calculate the tf.nn.softmax_cross_entropy_with_logits the passed logits shouldn't be softmax 
	#This op expects unscaled logits, since it performs a softmax on logits internally for efficiency
	with tf.name_scope('fc7') as scope: 
		W_sm7 = weight_variable([fc_units,num_labels])
		#W_sm7 = variable_with_weight_decay([fc_units,num_labels],stddevi, wdecay)
		b_sm7 = bias_variable([num_labels])
		logits=tf.matmul(h_fc6, W_sm7) + b_sm7
	  
	#with tf.name_scope('softmax7') as scope: 
	#    W_sm7 = weight_variable([fc_units,num_labels])
	#    b_sm7 = bias_variable([num_labels])
	#    logits=tf.nn.softmax(tf.matmul(h_fc6, W_sm7) + b_sm7)
  
	return logits
	
