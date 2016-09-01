import numpy as np
import os
from PIL import Image
import glob
from sklearn import metrics
from gco_python.pygco import cut_simple, cut_from_graph
from matplotlib import cm

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

def smoothing_oct(unaries):
    # potts potential
    label_num = unaries.shape[2]
    x = np.argmin(unaries, axis=2)
    # potential that penalizes 0-1 and 1-2 less thann 0-2
    A = -1000
    B = 800
    pairwise_1d = A * np.eye(label_num, dtype=np.int32) - B
    pairwise_1d[3, 5] = 0
    pairwise_1d[5, 3] = 0
    pairwise_1d[0, 5] = 0
    pairwise_1d[5, 0] = 0
    pairwise_1d[1, 5] = 0
    pairwise_1d[5, 1] = 0
    pairwise_1d[0, 3] = 0
    pairwise_1d[3, 0] = 0
    pairwise_1d[1, 4] = 0
    pairwise_1d[4, 1] = 0
    pairwise_1d[0, 1] = 0
    pairwise_1d[1, 0] = 0
    pairwise_1d[2, 1] = -B/2
    pairwise_1d[1, 2] = -B/2
    pairwise_1d
    return cut_simple(unaries, pairwise_1d)


def getCenterLabel(pixel):
    """
    outside = red = (255,0,0)
    inside = green = (255,255,0)
    lumen = yellow = (0,255,0)
    catheder shadow = blue = (0,0,255)
    artery wall = turquisw = (0,255,255)
    stent = pink = (255,0,255)
    
    input: RGB value of pixel
    output: integer between 1-5 represnting a class described above
    """
    pixel = tuple(pixel)
    rgb2label = {(255,0,0):0, (255,255,0):1, (0,255,0):2, 
                    (0,0,255):3, (0,255,255):4, (255,0,255):5}
    return rgb2label.get(pixel, 6)

def crop_image(image, size):
    '''Crop sized centre of image  
    
    :param PIL image image: image to be cropped
    :return: cropped image
    :rtype: PIL image
    '''
    factor = (1-size)/2
    width, height = image.size
    box = factor*width,factor*height,(1-factor)*width,(1-factor)*height
    return image.crop(box)



#cwd_eval = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/resulst_from_flow/'
#cwd_data = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Data_all/Multiple/'
cwd_eval = '../Data/Evaluationdata/'
cwd_data = '../Data/Evaluationdata/Multiple/'


 

for box_size in [12,18,24,30,36,42,48,54,60,66]:
    f_cnn = open(cwd_eval+'F1_scores2_cnn%s.txt'%str(box_size), 'w')
    f_cnn.write('F1 scores%s \n' % str(box_size))
    f_cnn.write('image, f1score 0, f1score 1, f1score 2, f1score 3, f1score 4, f1score 5, f1score 6\n')

    f_cnn_em = open(cwd_eval+'F1_scores2_cnn_em%s.txt'%str(box_size), 'w')
    f_cnn_em.write('F1 scores%s \n' % str(box_size))
    f_cnn_em.write('image, f1score 0, f1score 1, f1score 2, f1score 3, f1score 4, f1score 5, f1score 6\n')

    print('Box_Size:',str(box_size))
    for image in ['01','06','12','18','24','30','36','42','48','54','60']:
        print('Image',image)
        data_gt_name = 'Image0%s.png'%image
        data_cnn_name = 'Image0%s_cnn_1_%s.png'%(image, str(box_size))
        data_cnn_em_name = 'Image0%s_cnn_em_1_%s.png'%(image, str(box_size))

        dataset = {}
        dataset['im_gt'] = np.asarray(crop_image(Image.open(cwd_data + data_gt_name),0.6))
        dataset['im_cnn'] = np.asarray(crop_image(Image.open(cwd_eval + data_cnn_name),0.6))
        dataset['im_cnn_em'] = np.asarray(apply_em(crop_image(Image.open(cwd_eval + data_cnn_name),0.6))) #np.asarray(crop_image(Image.open(cwd_eval + data_cnn_em_name),0.6))

        for data in dataset:
            dataset[data] = dataset[data].reshape(-1,3)
            dataset[data] = [getCenterLabel(dataset[data][i]) for i in xrange(len(dataset[data]))]

        f1_score_cnn = []
        f1_score_cnn_em = []
        f1_score_cnn = metrics.f1_score(dataset['im_gt'], dataset['im_cnn'], labels = [0,1,2,3,4,5], average= None)
        f1_score_cnn_em = metrics.f1_score(dataset['im_gt'], dataset['im_cnn_em'], labels = [0,1,2,3,4,5], average= None)

        f_cnn.write(image+',')
        for score in f1_score_cnn:
            f_cnn.write(str(score)+',')
        f_cnn.write('0.0\n')

        f_cnn_em.write(image+',')
        for score in f1_score_cnn_em:
            f_cnn_em.write(str(score)+',')
        f_cnn_em.write('0.0\n')

    f_cnn.close()
    f_cnn_em.close()






