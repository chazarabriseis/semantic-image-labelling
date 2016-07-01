
# prepare data from one image produce smaller iamges to be trained on labelling 

from PIL import Image
import tensorflow
import glob
import os
import numpy as np
import pickle
import timeit
from random import shuffle


# ### Set the directories of the data
cwd_gts = '/u/juliabal/OCT-project/Data/Trainingdata/Binary/'
cwd_top = '/u/juliabal/OCT-project/Data/Trainingdata/Raw/'
cwd_data = '/u/juliabal/OCT-project/Data/'
os.chdir(cwd_gts)
os.getcwd()

box_size = 12
pickle_file = 'size%s-balance.pickle' % str(box_size)

data_top = sorted(glob.glob('*'))
print data_top


def getDominantLabel(im_gst_crop):
    """"Is counting occurancy of all different labels in the image and labels it with the most dominant one
    """
    #Convert the image into an array with RGB pixel values
    im_gst_pixel = im_gst_crop.load()  # this is not a list, nor is it list()'able
    width, height = im_gst_crop.size
    all_pixels = []
    for x in range(width):
        for y in range(height):
            all_pixels.append(im_gst_pixel[x,y])
    #Counts the number opf occurences of each colour
    counts = np.zeros(2)
    #all the black pixels with color (0,0,0)
    counts[0] = all_pixels.count((0,0,0))
    counts[1] = (width*height)-counts[0]

    counts = counts.tolist()
    if counts[1]/(width*height) > 0.4:
        return 1
    else: return 0
    #return counts.index(max(counts))
    


### Set the parameter for data preparation
#define the box size around 
max_box_size = 48
stride = 10
print "%s images will be created per image" % str(((((900-max_box_size)/stride)+1) * (((900-max_box_size)/stride)+1)) )
print "%s images will be created of the entire training set" % str(((((900-max_box_size)/stride)+1) * (((900-max_box_size)/stride)+1))*len(data_top))


### Load the ground truth and image data and cut the into images with 64x64 images and the according label
image_data_balance = []
label_pixel_balance = []
for item in data_top:
    image_data = []
    label_pixel = []
    index_1 = []
    index_0 = []
    im_gst = Image.open(cwd_gts + item)
    im_top = Image.open(cwd_top + item)
    if (im_gst.size != im_top.size):
        print "Problem, gst and top data doesn't match"
    #Define the iwdth and the height of the image to be cut up in smaller images
    width, height = im_gst.size
    #Since the outer image doesn't have real information the image is cropped first to a smaller box
    box = 0.2*width,0.2*height,0.8*width,0.8*height
    im_gst = im_gst.crop(box)
    im_top = im_top.crop(box)
    width, height = im_gst.size
    print "Image widht %s and image height %s" % (width, height) 
    #Go through the height (y-axes) of the image
    for i in range(0,(int((height- max_box_size)/stride) +1)):
        center_point_y = max_box_size/2+i*stride
        #Go through the width (x-axes) of the image using the same centerpoint independent of boxsize
        for j in range(0,(int((width- max_box_size)/stride) + 1)):
            center_point_x = int(max_box_size/2+j*stride)
            box = int(center_point_x-box_size/2), int(center_point_y-box_size/2), int(center_point_x+box_size/2), int(center_point_y+box_size/2)
	    image_data.append(np.array(im_top.crop(box)))
            label_pixel.append(getDominantLabel(im_gst.crop(box)))
    #Here we create a balnaced dataset of 50% stent labels (label 1) and 50% others (label 0)
    #create two lists with all the indices with label 1 or label 0
    index_1 = [i for i, j in enumerate(label_pixel) if j == 1]
    index_0 = [i for i, j in enumerate(label_pixel) if j == 0]
    #shuffle the index list of the 0 labels
    shuffle(index_0)
    #append all stent labelled images and it's labels 1
    for i in range(0,len(index_1)):
        image_data_balance.append(image_data[index_1[i]])
        label_pixel_balance.append(1)
        #append as many other labelled images as stent labelled images exist (the shuffling makes sure they are randomly others)
        image_data_balance.append(image_data[index_0[i]])
        #[image_data[i] for i in index_0[0:len(index_1)]])
        label_pixel_balance.append(0)   
    print 'Processed %s and added %d images'%(item, len(index_1))


print len(label_pixel_balance)


# IConvert images in a np.array
labels = np.asarray(label_pixel_balance)
dataset = np.asarray(image_data_balance)

# Randomise a dataset together with the labels
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(dataset,labels)


# ### Save the images in a pickle file for later use
os.chdir(cwd_data)

try:
  f = open(pickle_file, 'wb')
  save = {
    'dataset': train_dataset,
    'labels': train_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)



