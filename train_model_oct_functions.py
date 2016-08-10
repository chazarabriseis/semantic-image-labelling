from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import pickle
import numpy as np


def load_data(pickle_file):
	'''Loads data prepared for this model from a pickle file
	
	:param str pickle_file: name of file to be loaded
	:param str cwd: path to file
	:return: images: images with shape (data_size, input_size, input_size, 3)
	:rtype: np.float32
	:return: labels: labels with shape (data_size, num_labels)
	:rtype: np.float32
	'''
	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		images = save['images']
		labels = save['labels']
		del save  # hint to help gc free up memory
		print('All images and labels', images.shape, labels.shape)
	return images, labels


def data_prep(images, labels, v_size = 0.0, te_size = 0.0):
	'''Splits data into training, validation and test data and applies 1-hot encoding to labels

	:param np.float32 images: RGB images with shape (data_size, input_size, input_size, 3)
	:param np.float32 labels: labels with shape (data_size, )
	:param float v_size: percentage of validation set (1 = 100%)
	:param float te_size: percentage of test set (1 = 100%)
	:return: images: train, validation and test images with shape (data_size, input_size, input_size, 3)
	:rtype: np.float32
	:return: labels: train, validation and test labels with shape (data_size, num_labels)
	:rtype: np.float32
	'''
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
	print('Created training set', train_images.shape, train_labels.shape)
	print('Created validation set', valid_images.shape, valid_labels.shape)
	print('Created test set', test_images.shape, test_labels.shape)
	return train_images, train_labels, valid_images, valid_labels, test_images, test_labels


def get_kernel_size(box_size):
	'''Set kernel size according to box_size
    
	:param int box_size: image size to train on
	:return: kernel size for this model
	:rtype: int
	'''
	box_size2kernel = {24:3, 30:3, 36:3, 42:5, 48:5, 54:5, 60:5}
	return box_size2kernel.get(box_size, 5)


def train_model(images,labels,input_size,kernel_size,cwd_data,cwd_checkpoint,run_name,num_epoch=40,num_labels=6):
	'''Trains a CNN network and saves the trained model in cwd_checkpoint path

	:param np.float32 images: RGB images with shape (training_size, input_size, input_size, 3)
	:param np.float32 labels: labels with shape (training_size, num_labels)
	:param int input_size: width and height of images
	:param int kernel_size: kernel size of network
	:param str cwd_data: path to data folder
	:param str cwd_data: path to checkpoint folder
	:param int num_epoch: number of epochs model should train for
	:param int num_labels: number of classes the network trains for
	'''

	# Real-time data preprocessing
	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	# Real-time data augmentation
	img_aug = ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=360.)
	img_aug.add_random_blur(sigma_max=3.)
	img_aug.add_random_flip_updown()

	# Convolutional network building
	network = input_data(shape=[None, input_size, input_size, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
	network = conv_2d(network, input_size/2, kernel_size, activation='relu')
	network = max_pool_2d(network, 2)
	network = conv_2d(network, input_size, kernel_size, activation='relu')
	network = max_pool_2d(network, 2)
	network = conv_2d(network, input_size*2, kernel_size, activation='relu')
	network = max_pool_2d(network, 2)
	network = conv_2d(network, input_size*2*2, kernel_size, activation='relu')
	network = max_pool_2d(network, 2)
	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.5)
	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.5)
	network = fully_connected(network, num_labels, activation='softmax')
	network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

	# Train using classifier
	model = tflearn.DNN(network, tensorboard_verbose=0,tensorboard_dir=cwd_data,checkpoint_path=cwd_checkpoint,max_checkpoints=2)
	#model.load(cwd_data+'oct-cvn-48bal-6c-114300')
	model.fit(images, labels, n_epoch=num_epoch, validation_set=0.1, show_metric=True, run_id=run_name, snapshot_epoch=True)




