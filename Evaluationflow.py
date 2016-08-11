import glob
from PIL import Image
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from evaluate_model_oct_functions import get_model_name, load_image, label_on_fly, make_image, apply_em, save_image, create_f1_score, get_kernel_size, get_label_from_cnn, open_file
from prepare_data_oct_functions import crop_image

#Set parameter of model to be evaluated

#cwd_raw = '../Data/Evaluationdata/Raw/'
#cwd_label = '../Data/Evaluationdata/Multiple/'
#cwd_data = '../Data/Evaluationdata/'
#cwd_model = '../Data/Checkpoints'
cwd_raw = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Evaluationdata/Raw/'
cwd_label = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Evaluationdata/Multiple/'
cwd_data = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Evaluationdata/'
cwd_model = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/final/'

stride = 100
box_size = 48
model_name = get_model_name(cwd_model,box_size)

data_eval = sorted(glob.glob1(cwd_raw,'*.jpg*'))
print data_eval

#Open file for f1 score
f = open(cwd_data+'F1_scores%s.txt'%str(box_size), 'w')
f.write('F1 scores%s \n' % str(box_size))
f.write('Box_size, f1score 0, f1score 1, f1score 2, f1score 3, f1score 4, f1score 5, f1score overall\n')
    
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
model = tflearn.DNN(network, tensorboard_verbose=0,tensorboard_dir=cwd_model,checkpoint_path=cwd_model)
print('Loading model:', model_name)
model.load(model_name)
print('Sucessfully loaded model for prediction')

for data_name in data_eval:

    data_name_label = cwd_label+data_name[:data_name.find('.jpg')] + '.png'
    prediction_cnn_name =  cwd_data+data_name[:data_name.find('.jpg')]+'_cnn_'+str(stride)+'_'+str(box_size)+ '.png'
    prediction_cnn_em_name =  cwd_data+data_name[:data_name.find('.jpg')]+'_cnn_em_'+str(stride)+'_'+str(box_size)+ '.png'

    ##Load evaluation image and prepare for evaluation
    im = load_image(cwd_raw+data_name)
    size = im.size
    print("####Succesfully loaded image",data_name)

    ##load the model and calculate predictions for each pixel
    max_box_size = box_size
    labels_predcited = []
    #Define the width and the height of the image to be cut up in smaller images
    im = crop_image(im, 0.6)
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
    labels_cnn = labels_final
    print("Succesfully calculated cnn predictions")

    ##create an output image of the raw predictions
    im_prediction_cnn = make_image(labels_cnn, size)
    print("Succesfully created cnn prediction image")

    ## Save the raw prediction
    save_image(im_prediction_cnn, prediction_cnn_name)
    print("Succesfully saved cnn prediction image")

    ##improve prediction image by applying energy minimization
    im_prediction_cnn_em = apply_em(im_prediction_cnn)
    print("Succesfully calculated prediction using EM")

    ## Save the final prediction
    save_image(im_prediction_cnn_em, prediction_cnn_em_name)
    print("Succesfully saved cnn+em prediction image")

    ##Save the F1 scores in a txt file
    im_prediction_gt = load_image(data_name_label)
    #write into f1 score
    f.write(data_name[5:data_name.find('.jpg')]+',')
    f1_scores = create_f1_score(im_prediction_gt,im_prediction_cnn)
    for score in f1_scores:
        f.write(str(score)+',')
    f.write('0.0\n')

#close file for f1 score
f.close()