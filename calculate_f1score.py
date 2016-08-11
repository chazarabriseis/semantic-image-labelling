import numpy as np
import os
from PIL import Image
import glob
from sklearn import metrics

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



cwd_eval = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/resulst_from_flow/'
cwd_data = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Data_all/Multiple/'
#cwd_eval = '..//Data/resulst_from_flow/'
#cwd_data = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Data_all/Multiple/'


 

for box_size in [12,18]:
    f_cnn = open(cwd_eval+'F1_scores_cnn%s.txt'%str(box_size), 'w')
    f_cnn.write('F1 scores%s \n' % str(box_size))
    f_cnn.write('image, f1score 0, f1score 1, f1score 2, f1score 3, f1score 4, f1score 5, f1score 6\n')

    f_cnn_em = open(cwd_eval+'F1_scores_cnn_em%s.txt'%str(box_size), 'w')
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
        dataset['im_cnn_em'] = np.asarray(crop_image(Image.open(cwd_eval + data_cnn_em_name),0.6))

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






