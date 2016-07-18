
# tf_learn script to train a CVNN for image pixel labelling

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import semantic_functions as sf

import os

box_size = 48
kernel_size = 5
epoch_nummer = 50
pickle_file = 'size%s-6c-balance.pickle' % str(box_size)
run_name = 'oct-cvn-%sbal-6c' % str(box_size)
cwd_checkpoint = '/u/juliabal/OCT-project/Data/Checkpoints/%s' % run_name
cwd = '/u/juliabal/OCT-project/'
cwd_data = '/u/juliabal/OCT-project/Data/'
os.chdir(cwd_data)

#Load Input Data from 
images, labels = sf.load_data(pickle_file, cwd_data)
X, Y, X_valid, Y_valid, X_test, Y_test = sf.data_prep(images, labels, 0, 0)

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

input = box_size
# Convolutional network building
network = input_data(shape=[None, input, input, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, input/2, kernel_size, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, input, kernel_size, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, input*2, kernel_size, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, input*2*2, kernel_size, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 6, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0,tensorboard_dir=cwd,checkpoint_path=cwd_checkpoint,max_checkpoints=10)
model.fit(X, Y, n_epoch=epoch_nummer, validation_set=0.1, show_metric=True, run_id=run_name, snapshot_epoch=True)




