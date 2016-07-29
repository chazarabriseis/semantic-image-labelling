from __future__ import division, print_function, absolute_import
from PIL import Image
import glob
import os
import numpy as np
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

def loadImage(image_name):
    """
    Loads an image from a certain directory

    Input: image name, path

    Output: PIL image
    """
    return Image.open(image_name)

def labelOntheFly(pic, model_name,cwd_checkpoint, stride, box_size):
    """
    Takes and image and prepares it in the same way the images 
    were prepared for training the model which will be used to predict a label
    Goes through the image line by line to avoid using too much 

    Input: PIL image for prediction

    Output: np.array that stores sections of the image for prediction 
            (depends on the model for predictions)
    """
    input_size = box_size
    kernel_size = 5
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
    network = fully_connected(network, 6, activation='softmax')
    network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
    # Defining model
    model = tflearn.DNN(network, tensorboard_verbose=0,tensorboard_dir=cwd_checkpoint,checkpoint_path=cwd_checkpoint)
    model.load(model_name)
    print('Model sucessfully loaded for label on the fly')
    
    max_box_size = box_size
    #stride = 1
    labels_predcited = []
    #Define the width and the height of the image to be cut up in smaller images
    width, height = pic.size
    box = 0.2*width,0.2*height,0.8*width,0.8*height
    pic = pic.crop(box)
    width, height = pic.size
    #Go through the height (y-axes) of the image
    for i in range(0,int((height- max_box_size)/stride +1)):
        center_point_y = max_box_size/2+i*stride
        pic_temp = []
        #Go through the width (x-axes) of the image using the same centerpoint independent of boxsize
        for j in range(0,int((width- max_box_size)/stride + 1)):
            center_point_x = max_box_size/2+j*stride
            box = center_point_x-box_size/2, center_point_y-box_size/2, center_point_x+box_size/2,center_point_y+box_size/2
            pic_temp.append(pic.crop(box))
        predictions_temp = model.predict(pic_temp)
        labels_temp = [get_label(predictions_temp[i]) for i in range(0,len(predictions_temp))]
        labels_predcited.append(labels_temp)
    labels_final = [item for i in range(0,len(labels_predcited)) for item in labels_predcited[i]]
    print(labels_final[0:2])
    return(labels_final)

def get_label(prediction):
    """
    Convert the label into a rgb colour value
    
    Input: Vector with 1-hot encoding labels
    
    Output: Vector with rgb colour
    """
    index = prediction.index(max(prediction))
    return colorizer(index)

def colorizer(pixel):
    """
    background = black = (0,0,0)
    outside = red = (255,0,0)
    inside = green = (255,255,0)
    lumen = yellow = (0,255,0)
    catheder shadow = blue = (0,0,255)
    artery wall = turquisw = (0,255,255)
    stent = pink = (255,0,255)
    
    input integer between 0-5
    output RGV value of pixel
    """
    if pixel == 0:
        return (255,0,0)
    if pixel == 1:
        return (255,255,0)
    if pixel == 2:
        return (0,255,0)
    if pixel == 3:
        return (0,0,255)
    if pixel == 4:
        return (0,255,255)
    if pixel == 5:
        return (255,0,255)

def makeImage(predictions, box_size = 48):
    """
    Creates an image from the predictions and scales it to the original image

    Input: np.array with the predictions of each area

    output: PIL image
    """
    size = int(np.sqrt(len(predictions))), int(np.sqrt(len(predictions)))
    im_gts_rgb = predictions #[get_label(predictions[i]) for i in range(0,len(predictions))]
    im_label = Image.new("RGB", size)
    im_label.putdata(im_gts_rgb)
    #Here the image gets expaned top the original image size
    #im_label = im_label.transform((box_size*int(np.sqrt(len(predictions))),box_size*int(np.sqrt(len(predictions)))), Image.EXTENT, (0,0,int(np.sqrt(len(predictions))),int(np.sqrt(len(predictions)))))
    return im_label

def save_image(im_label, name):
    """
    Saves a PIL image
    
    Imput: Pil image, path to save image OPTIONAL, name of image file OPTIONAL
    
    Output: File in Path of image
    """
    im_label.save(name)

