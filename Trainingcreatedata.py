########################################################################################## 
####    Code to prepare OCT images for traininag a CNN for semantic image pixel labelling
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
####	Prepare the data

### Set parameters for data preparation
#box_size = int(sys.argv[1])
box_size = 48
pickle_file = '%ssize%s-6c-balance.pickle' % (cwd_data,str(box_size))

data_list = sorted(glob.glob1(cwd_label,'*034*'))
print data_list

### Prepare data for the CVN
images, labels = get_data(data_list, box_size, cwd_raw, cwd_label)
print 'Length of created dataset', len(labels)
images, labels = randomize(images,labels)
	
### Save them as a pickle file
save_pickeldata(images, labels, pickle_file)
