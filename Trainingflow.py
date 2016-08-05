########################################################################################## 
####	Code to train a CVN for semantic image pixel labelling
####	Author: Julia Baldauf
####	5.8.2016		

import glob
from prepare_data_oct_functions import get_data, randomize, save_pickeldata
from train_model_oct_functions import load_data, data_prep, train_model


########################################################################################## 
####	Set directories

#cwd_raw = '../Data/Trainingdata/Raw/'
#cwd_label = '../Trainingdata/Multiple/'
#cwd_data = '../Data/'
cwd_raw = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Trainingdata/Raw/'
cwd_label = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Trainingdata/Multiple/'
cwd_data = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/'


##########################################################################################
####	Prepare the data

### Set the parameters for data preparation
box_size = 48
pickle_file = '%ssize%s-6c-balance.pickle' % (cwd_data,str(box_size))

data_list = sorted(glob.glob1(cwd_label,'*'))
print data_list

### Prepare the data for the CVN
images, labels = get_data(data_list, box_size, cwd_raw, cwd_label)
print 'Length of created dataset', len(labels)
images, labels = randomize(images,labels)
	
### Save them as a pickle file
save_pickeldata(images, labels, pickle_file)


##########################################################################################
####	Train the model

### Set the parameters for training
box_size2kernel = {48: 5, 24:3, 12:2}
kernel_size = box_size2kernel[box_size]
run_name = 'oct-cvn-%sbal-6c' % str(box_size)
cwd_checkpoint = cwd_data+'Checkpoints/%s' % run_name
num_labels = 6
num_epoch = 40

### Load data & prepare the data
images, labels = load_data(pickle_file)
images, labels, _, _, _, _ = data_prep(images, labels)

train_model(images, labels, box_size, kernel_size, cwd_data, run_name, num_epoch, num_labels)
