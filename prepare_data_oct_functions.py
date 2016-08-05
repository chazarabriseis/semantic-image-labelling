from PIL import Image
import tensorflow
import glob
import os
import numpy as np
import pickle
import timeit
from random import shuffle
    

def get_center_label(pixel):
    '''
    outside = red = (255,0,0) = 0
    lumen = yellow = (255,255,0) = 1
    inside = green = (0,255,0) = 2
    catheder shadow = blue = (0,0,255) = 3
    artery wall = turquise = (0,255,255) = 4 
    stent = pink = (255,0,255) = 5
    
    input: RGB value of pixel
    output: integer between 1-5 representing a class described above
    '''
    rgb2label = {(255,0,0):0, (255,255,0):1, (0,255,0):2, 
                    (0,0,255):3, (0,255,255):4, (255,0,255):5}
    return rgb2label.get(pixel, 6)


def crop_image(image, factor):
    '''
    crop image by a factor
    
    Input: PIL image
    Output: PIL image
    '''
    width, height = image.size
    box = factor*width,factor*height,(1-factor)*width,(1-factor)*height
    return image.crop(box)


def make_balanced_dataset(images,labels,images_balanced,labels_balanced):
    '''
    Create a balanced dataset adjusting everything to the smallest class represented  

    Input: list of images, list of labels
    Output: balanced list of images and list of labels
    '''
    indices = {}
    for count in xrange(6):
        indices[count] = [i for i, j in enumerate(labels) if j == count]
    #shuffle the index list of the labels not being a stent to increase random sampling
    for index in indices:
        shuffle(indices[index])         
    #append all stent labelled images and the same amount of each other class
    #index_small = min(indices, key= lambda x: len(set(indices[x])))
    index_small = 5     #balance stent images
    length_small = len(indices[index_small])
    for index in indices:
        for item in indices[index][:length_small]:
            images_balanced.append(images[item])
            labels_balanced.append(index)  
    print 'Added %s images (label %s)' % (str(length_small), str(index_small))
    #make a note if catheter shadow pixels <  stent pixels for f1 score
    if len(indices[3]) < len(indices[5]):
        print 'Less catheter shadow images added', len(indices[3])
    return images_balanced, labels_balanced


def get_data(data_list, box_size, cwd_raw, cwd_label):
    '''
    Create a dataset with smaller images and their center pixel value as a label

    Input: file, size of the smaller image, path to labels of file, path to raw image of file
    Output: List of images, List of labels
    '''
    stride = 10
    images_balanced = []
    labels_balanced = []
    for data_name in data_list:
        print '####   Processing: ', data_name
        im_label = Image.open(cwd_label + data_name)
        pixels_label = im_label.load()
        im_raw = Image.open(cwd_raw + data_name[:data_name.find('.png')] + '.jpg')
        #assert (im_label.size != im_raw.size), "Problem, label and raw data don't have the same size"
        #Remove the outside of the images to concentre on the centre where all the action is happening
        factor =0.2
        im_label = crop_image(im_label,factor)
        im_raw = crop_image(im_raw,factor)
        #Go through the height (y-axes) of the image
        width, height = im_label.size
        images = [] 
        labels = []
        for i in xrange(int(height-box_size)/stride):
            center_point_y = box_size/2+i*stride
            #Go through the width (x-axes) of the image using the same centerpoint independent of boxsize
            for j in xrange(int(width-box_size)/stride):
                center_point_x = box_size/2+j*stride
                box = center_point_x-box_size/2, center_point_y-box_size/2, center_point_x+box_size/2,center_point_y+box_size/2
                images.append(np.array(im_raw.crop(box)))
                labels.append(get_center_label(pixels_label[center_point_x,center_point_y]))
        images_balanced, labels_balanced = make_balanced_dataset(images,labels,images_balanced,labels_balanced)
    return images_balanced, labels_balanced


def randomize(images, labels):
    '''
    randomisez a dataset and labels by keeping the order

    Input: list of images, list of labels
    Output: randomized list of images and list of labels
    '''
    images = np.asarray(images)
    labels = np.asarray(labels)
    permutation = np.random.permutation(labels.shape[0])
    shuffled_images = images[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_images, shuffled_labels


def save_pickeldata(train_images, train_labels, pickle_file):
    '''
    Saves data of images and labels for later use

    Input: list of images, list of labels, file to save it to
    '''
    try:
        f = open(pickle_file, 'wb')
        save = {
            'images': train_images,
            'labels': train_labels,
            }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)



