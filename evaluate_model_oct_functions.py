from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
from sklearn import metrics
from gco_python.pygco import cut_simple, cut_from_graph
from matplotlib import cm
import glob

def get_model_name(cwd_model,box_size):
    models = sorted(glob.glob1(cwd_model,'*oct-cvn-*'))
    model = models[[item for item, check in enumerate(models) if check.find(str(box_size)+'bal')>0][0]]
    return cwd_model+model

def open_file(box_size, cwd_data):
    f = open(cwd_data+'F1_scores%s.txt'%str(box_size), 'w')
    f.write('F1 scores%s \n' % str(box_size))
    f.write('Box_size, f1score 0, f1score 1, f1score 2, f1score 3, f1score 4, f1score 5, f1score overall\n')
    return f

def convert_to_colour(input_array):
    '''Converts array in image with oct colour code

    :param double input_array: array of image pixels
    :retrun: image with oct colour code
    :rtye: PIL image
    '''
    im = Image.fromarray(np.uint8(cm.rainbow(input_array)*255))
    x,y = im.size
    im_index = im.load()
    label_new = []
    for i in xrange(x):
        for j in xrange(y):
            rgb =  (im_index[j,i][0],im_index[j,i][1],im_index[j,i][2])
            label_new.append(get_label_from_rgb_em(rgb))
    im_new = Image.new('RGB',(x,y))
    im_new.putdata(label_new)
    return im_new


def vectorized_result(j,num_label):
    '''Convert number to 1-hot-encoding vector
    
    :param int j: label
    :param int num_label: number of classes determining length of vector
    :return: num_label-dimensional unit vector with a 1.0 in the jth
    :rtype: int
    '''
    e = np.zeros((num_label, 1))
    e[int(j)] = 1
    return e


def smoothing_oct(unaries):
    '''Apply energy minimization from http://peekaboo-vision.blogspot.com.au/2012/05/graphcuts-for-python-pygco.html

    :param int unaries: array of labels
    :return: smoothend labels
    :rtype: int
    '''
    label_num = unaries.shape[2]
    x = np.argmin(unaries, axis=2)
    pott_potential = -1000 * np.eye(label_num, dtype=np.int32)
    return cut_simple(unaries, pott_potential)

    
def apply_em(im_prediction_cnn):
    '''Smoothening image applying energy minimization

    :param PIL image im_prediction: image with prediction from cnn
    :return: smoothened image
    :rtype: PIL image
    '''
    print("Applying energy minimization")
    size = im_prediction_cnn.size
    labels_cnn = np.asarray(im_prediction_cnn).reshape(-1,3)
    labels_cnn = [vectorized_result(get_label_from_rgb(labels_cnn[i]),7) for i in xrange(len(labels_cnn))]
    labels_cnn = np.asarray(labels_cnn)
    labels_cnn = (labels_cnn.reshape(int(size[0]),int(size[1]),7)*(-100)).astype(np.int32)
    return convert_to_colour(smoothing_oct(labels_cnn))


def create_f1_score(im_prediction_gt,im_prediction_cnn):
    '''Calculating F1 score

    :param int im_prediction: rgb tuples representing ground truth labels
    :param int im_prediction_cnn: rgb tuples representing predicted labels
    :return: f1 score between the two labels
    :rtype: double
    '''
    print("Calculating f1 score")
    labels_gt = np.asarray(im_prediction_gt).reshape(-1,3)
    labels_cnn = np.asarray(im_prediction_cnn).reshape(-1,3)
    labels_cnn = [get_label_from_rgb(labels_cnn[i]) for i in xrange(len(labels_cnn))]
    labels_gt = [get_label_from_rgb(labels_gt[i]) for i in xrange(len(labels_gt))]
    score = metrics.f1_score(labels_cnn,labels_gt, average= None)
    return score

def get_label_from_rgb(pixel):
    '''Convert RGB pixel to label

    outside = red = (255,0,0) = 0
    lumen = yellow = (255,255,0) = 1
    inside = green = (0,255,0) = 2
    catheder shadow = blue = (0,0,255) = 3
    artery wall = turquise = (0,255,255) = 4 
    stent = pink = (255,0,255) = 5
    
    :param int pixel: tuple of RGB value of pixel
    :return: label representing a class described above
    :rtype: int
    '''
    pixel = tuple(pixel)
    rgb2label = {(255,0,0):0, (255,255,0):1, (0,255,0):2, 
                    (0,0,255):3, (0,255,255):4, (255,0,255):5}
    return rgb2label.get(pixel, 6)

    
def load_image(image_name):
    '''Load image 

    :param str image_name: path to image to be loaded
    :return: image 
    :rtype: PIL image
    '''
    return Image.open(image_name)

def get_kernel_size(box_size):
    '''Set kernel size according to box_size
    
    :param int box_size: image size to train on
    :return: kernel size for this model
    :rtype: int
    '''
    box_size2kernel = {24:3, 30:3, 36:3, 42:5, 48:5, 54:5, 60:5}
    return box_size2kernel[box_size]


def label_on_fly(im, model_name,cwd_checkpoint, stride, box_size):
    '''Converts pixels of image into labels

    Goes through smaller image and prepares it in the same way the images 
    were prepared for training the model which will be used to predict a label
    Goes through the image line by line to avoid using too much memory 

    :param PIL image im: for prediction, 
    :parammodel to load, 
    :param path to model, 
    :param stride of scanning image
    :param box_size
    :return: rgb values of the label for each pixel
    :rtype: int tuples
    '''
    input_size = box_size
    kernel_size = get_kernel_size(input_size)
    # Real-time data preprocessing
    im_prep = ImagePreprocessing()
    im_prep.add_featurewise_zero_center()
    im_prep.add_featurewise_stdnorm()
    # Real-time data augmentation
    im_aug = ImageAugmentation()
    im_aug.add_random_flip_leftright()
    im_aug.add_random_rotation(max_angle=360.)
    im_aug.add_random_blur(sigma_max=3.)
    im_aug.add_random_flip_updown()
    # Convolutional network building
    network = input_data(shape=[None, input_size, input_size, 3],
                     data_preprocessing=im_prep,
                     data_augmentation=im_aug)
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
    print('Loading model:', model_name)
    model.load(model_name)
    print('Sucessfully loaded model')
    
    max_box_size = box_size
    labels_predcited = []
    #Define the width and the height of the image to be cut up in smaller images
    width, height = im.size
    box = 0.2*width,0.2*height,0.8*width,0.8*height
    im = im.crop(box)
    width, height = im.size
    #Go through the height (y-axes) of the image
    for i in xrange(int((height-max_box_size)/stride)):
        center_point_y = max_box_size/2+i*stride
        im_temp = []
        predictions_temp = []
        labels_temp = []
        #Go through the width (x-axes) of the image using the same centerpoint independent of boxsize
        for j in xrange(int((width- max_box_size)/stride)):
            center_point_x = max_box_size/2+j*stride
            box = center_point_x-box_size/2, center_point_y-box_size/2, center_point_x+box_size/2,center_point_y+box_size/2
            im_temp.append(im.crop(box))
        predictions_temp = model.predict(im_temp)
        #labels_temp = [colorizer(get_label_from_cnn(predictions_temp[k])) for k in xrange(len(predictions_temp))]
        labels_temp = [get_label_from_cnn(predictions_temp[k]) for k in xrange(len(predictions_temp))]
        labels_predcited.append(labels_temp)
        print('Line %s done' % str(i))
    labels_final = [item for m in xrange(len(labels_predcited)) for item in labels_predcited[m]]
    return(labels_final)

def get_label_from_cnn(prediction):
    ''''Convert the label-vector from the CNN into an integer label
    
    :param float prediction: label vector with probabilties for wach class
    :return: label
    :rtype: int
    '''
    return prediction.index(max(prediction))

def get_label_from_rgb_em(pixel):
    '''Convert RGB pixel to label

    outside = red = (255,0,0) = 0
    lumen = yellow = (255,255,0) = 1
    inside = green = (0,255,0) = 2
    catheder shadow = blue = (0,0,255) = 3
    artery wall = turquise = (0,255,255) = 4 
    stent = pink = (255,0,255) = 5
    
    :param int pixel: tuple of RGB value of pixel
    :return: label representing a class described above
    :rtype: int
    '''
    pixel = tuple(pixel)
    rgb2label = {(115, 18, 254):(0,0,0), (127, 0, 255):(255,0,0), (123, 6, 254):(0,255,0), 
                    (119, 12, 254):(0,255,255), (125, 3, 254):(255,255,0), 
                    (117, 15, 254):(255,0,255), (121, 9, 254):(0,0,255)}
    return rgb2label[pixel]

def colorizer(label):
    '''Convert label to RGB pixel

    outside = red = (255,0,0) = 0
    lumen = yellow = (255,255,0) = 1
    inside = green = (0,255,0) = 2
    catheder shadow = blue = (0,0,255) = 3
    artery wall = turquise = (0,255,255) = 4 
    stent = pink = (255,0,255) = 5
    
    :param int label: label representing a class described above
    :return: RGB pixel corresponding to the label
    :rtype: tuple
    '''
    label2rgb = {0: (255,0,0), 1: (255,255,0), 2: (0,255,0), 
                    3: (0,0,255), 4: (0,255,255), 5: (255,0,255)}
    return label2rgb[label]


def make_image(predictions, size_orginal):
    '''Creates an image from the list of predictions and scales it to the original image

    :param float predictions_rgb: np.array with the predictions of pixels
    :param int size_orginal: tuple of hwight and width of original images
    :return: image of the predictions
    :rtype: PIL image
    '''
    predictions_rgb= [colorizer(predictions[k]) for k in xrange(len(predictions))]
    size = int(np.sqrt(len(predictions_rgb))), int(np.sqrt(len(predictions_rgb)))
    im = Image.new("RGB", size)
    im.putdata(predictions_rgb)
    #Here the image gets expaned to the size of the labele image size if stride was > 1
    size_training = (int(0.6*size_orginal[0]),int(0.6*size_orginal[1]))
    if size != size_training:
        im = im.transform(size_training, Image.EXTENT, (0,0,size[0],size[1]))
    size = im.size
    #Here the predicted image get framed so its size ist the one of the original input image
    im_label = Image.new("RGB", size_orginal)
    im_label.paste(im, (int((size_orginal[0]-size[0])/2),
                      int((size_orginal[1]-size[1])/2)))
    return im_label


def save_image(im_label, name):
    '''Saves a PIL image
    
    :param PILimage im_label: images to be saved
    :param str name: path for image to be saved
    '''
    im_label.save(name)


