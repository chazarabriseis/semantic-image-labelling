########################################################################################## 
####    Code to train a CNN for semantic image pixel labelling of OCT images
####    Author: Julia Baldauf
####    5.8.2016	
import glob
from prepare_data_oct_functions import get_data, randomize, save_pickeldata
from train_model_oct_functions import load_data, data_prep, train_model, get_kernel_size
import sys

########################################################################################## 
####	Set directories

cwd_raw = '../Data/Trainingdata/Raw/'
cwd_label = '../Data/Trainingdata/Multiple/'
cwd_data = '../Data/'
#cwd_raw = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Trainingdata/Raw/'
#cwd_label = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Trainingdata/Multiple/'
#cwd_data = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/'


##########################################################################################
####	Train the model

### Set parameters for training
box_size = 54
pickle_file = '%ssize%s-6c-balance.pickle' % (cwd_data,str(box_size))
kernel_size = get_kernel_size(box_size)
run_name = 'oct-cvn-run2-%sbal-6c' % str(box_size)
cwd_checkpoint = cwd_data+'Checkpoints/%s' % run_name
num_labels = 6
num_epoch = 50

### Load data & prepare the data
images, labels = load_data(pickle_file)
images, labels, _, _, _, _ = data_prep(images, labels)

train_model(images, labels, box_size, kernel_size, cwd_data, cwd_checkpoint, run_name, num_epoch, num_labels)

