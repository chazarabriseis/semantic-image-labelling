## Takes an OCT image
##		- load an image of interest
##		- prepares it to be feed in prediction
##		- load a trained model
##		- get the predictions
##		- transform predictions in a labelled image
##		- plots the prediction image

import numpy as np
import oct_functions_ccc as oct

stride = 100
#cwd_data = '../Data/Evaluationdata/Raw/'
#cwd_model = '../Data/Checkpoints/'
#cwd_save = '../Data/Evaluationdata/'
cwd_data = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Evaluationdata/Raw/'
cwd_model = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/final/'
cwd_save = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Evaluationdata/'

name = 'Image027.jpg'
evaldata_name = cwd_data+name
model_name = cwd_model+'oct-cvn-48bal-6c-114300'
prediction_name=cwd_save+ str(stride)+name
##from my local machine...for this example I set it to a knwon image: 'Image061.jpg' 
pic = oct.loadImage(evaldata_name)
print("Succesfully loaded image")

##load the model of interest and calculate predictions
pic_label = oct.labelOntheFly(pic,model_name,cwd_model,stride=stride,box_size = 48)
print("Succesfully calculated predictions")

##create an output image of the predictions
pic_prediction = oct.makeImage(pic_label)
print("Succesfully created prediction image")

## Plot this image next to the original
oct.save_image(pic_prediction, prediction_name)
print("Succesfully saved prediction image")
