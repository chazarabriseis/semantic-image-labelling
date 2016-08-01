
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
cwd_gts = '/u/juliabal/OCT-project/Data/Trainingdata/Binary2/'
cwd_top = '/u/juliabal/OCT-project/Data/Trainingdata/Raw/'
cwd_data = '/u/juliabal/OCT-project/Data/'
os.chdir(cwd_gts)
os.getcwd()

box_size = 48
pickle_file = '%ssize%s-6c-balance.pickle' % (cwd_data,str(box_size))

data_top = sorted(glob.glob('*'))
print data_top


def getDominantLabel(im_gst_crop):
    """"Is counting occurancy of all different labels in the 
    image and labels it with the most dominant one
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
    
def getCenterLabel(pixel):
    """
    background = black = (0,0,0)
    outside = red = (255,0,0)
    inside = green = (255,255,0)
    lumen = yellow = (0,255,0)
    catheder shadow = blue = (0,0,255)
    artery wall = turquisw = (0,255,255)
    stent = pink = (255,0,255)
    
    input: RGB value of pixel
    output: integer between 1-5 represnting a class described above
    """
    rgb2label = {(255,0,0):0, (255,255,0):1, (0,255,0):2, 
                    (0,0,255):3, (0,255,255):4, (255,0,255):5}
    return rgb2label.get[pixel, 6]


### Set the parameter for data preparation
#define the box size around 
max_box_size = 48
stride = 5
print "%s images will be created per image" 
            % str(((((900-max_box_size)/stride)+1) * (((900-max_box_size)/stride)+1)) )
print "%s images will be created of the entire training set" 
            % str(((((900-max_box_size)/stride)+1) * (((900-max_box_size)/stride)+1))*len(data_top))


### Load the ground truth and image data and cut the into images with 64x64 images and the according label
image_data_balance = []
label_pixel_balance = []
for item in data_top:
    image_data = []
    label_pixel = []
    im_gst = Image.open(cwd_gts + item)
    im_top = Image.open(cwd_top + item[:item.find('.png')] + '.jpg')
    if (im_gst.size != im_top.size):
        print "Problem, gst and top data doesn't match"
    #Define the width and the height of the image to be cut up in smaller images
    width, height = im_gst.size
    box = 0.2*width,0.2*height,0.8*width,0.8*height
    im_gst = im_gst.crop(box)
    im_top = im_top.crop(box)
    width, height = im_gst.size
    pixels_gst = im_gst.load()
    #Go through the height (y-axes) of the image
    for i in xrange(int((height- max_box_size)/stride +1)):
        center_point_y = max_box_size/2+i*stride
        #Go through the width (x-axes) of the image using the same centerpoint independent of boxsize
        for j in xrange(int((width- max_box_size)/stride + 1)):
            center_point_x = max_box_size/2+j*stride
            box = center_point_x-box_size/2, center_point_y-box_size/2, center_point_x+box_size/2,center_point_y+box_size/2
            image_data.append(np.array(im_top.crop(box)))
            label_pixel.append(getCenterLabel(pixels_gst[center_point_x,center_point_y]))
    #Here we create a balnaced dataset of 50% stent labels (label 1) and 50% others (label 0)
    #create two lists with all the indices with label 1 or label 0
    index_5 = [i for i, j in enumerate(label_pixel) if j == 5]
    index_4 = [i for i, j in enumerate(label_pixel) if j == 4]
    index_3 = [i for i, j in enumerate(label_pixel) if j == 3]
    index_2 = [i for i, j in enumerate(label_pixel) if j == 2]
    index_1 = [i for i, j in enumerate(label_pixel) if j == 1]
    index_0 = [i for i, j in enumerate(label_pixel) if j == 0]
    #shuffle the index list of the labels not being a stent
    shuffle(index_4)    
    shuffle(index_3)    
    shuffle(index_2)    
    shuffle(index_1)    
    shuffle(index_0)                           
    #append all stent labelled images and it's labels 1
    for i in range(0,len(index_5)):
        image_data_balance.append(image_data[index_5[i]])
        label_pixel_balance.append(5)
        #append as many other labelled images as stent labelled images exist (the shuffling makes sure they are randomly others)
        if index_4[i]:
            image_data_balance.append(image_data[index_4[i]])
            label_pixel_balance.append(4) 
        if i < len(index_3):
            image_data_balance.append(image_data[index_3[i]])
            label_pixel_balance.append(3) 
        if i < len(index_2):
            image_data_balance.append(image_data[index_2[i]])
            label_pixel_balance.append(2) 
        if i < len(index_1):
            image_data_balance.append(image_data[index_1[i]])
            label_pixel_balance.append(1) 
        if i < len(index_0):
            image_data_balance.append(image_data[index_0[i]])
            label_pixel_balance.append(0)   
    print 'Artery wall: %s Catheter shadow: %s Inside: %s Lumen: %s Outside: %s' % (len(index_4),len(index_3),len(index_2),len(index_1),len(index_0))
    print 'Processed %s and added %d images'%(item, len(index_5))


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



