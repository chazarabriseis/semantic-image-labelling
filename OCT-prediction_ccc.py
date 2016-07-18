## Takes an OCT image
##		- load an image of interest
##		- prepares it to be feed in prediction
##		- load a trained model
##		- get the predictions
##		- transform predictions in a labelled image
##		- plots the prediction image

import numpy as np
import oct_functions as oct

##from my local machine...for this example I set it to a knwon image: 'Image061.jpg' 
pic = oct.loadImage()
print("Succesfully loaded image")

##pass this picture to a function that prepares it for th 64pixel CVN
pic_top = oct.prepareImage(pic)
print("Succesfully prepared image for prediction")

##load the model of interest and calculate predictions
pic_label = oct.loadModel64(pic_top)
print("Succesfully calculated predictions")

##create an output image of the predictions
pic_prediction = oct.makeImage(pic_label)
print("Succesfully created prediction image")

## Plot this image next to the original
oct.save_image(pic_prediction, name = 'test.jpg')
print("Succesfully saved prediction image")
