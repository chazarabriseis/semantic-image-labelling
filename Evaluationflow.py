import glob
import os


#Set parameter of model to be evaluated
stride = 1
model_name = cwd_model+'oct-cvn-48bal-6c-114300'
box_size = int(model_name[model_name.find('cvn-')+4:model_name.find('bal')])

#cwd_raw = '../Data/Evaluationdata/Raw/'
#cwd_model = '../Data/Checkpoints/'
#cwd_save = '../Data/Evaluationdata/'
cwd_raw = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Evaluationdata/Raw/'
cwd_binary = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Evaluationdata/Binary2/'
cwd_model = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/final/'
cwd_save = '/Users/jbaldauf/Documents/Tensorflow/OCT-project/Data/Evaluationdata/'


data_eval = sorted(glob.glob('*png*'))

for data_name in data_eval:

	##Set names of created images
	evaldata_raw_name = cwd_raw+data_name
	evaldata_gt_name = cwd_binary+data_name
	prediction_cnn_name =  cwd_save+'_cnn_'+name[:name.find('jpg')]+'png'
	prediction_cnn_em_name =  cwd_save+'_cnn_em_'+name[:name.find('jpg')]+'png'

	##prepare the data from evaluationdata
	pic = oct.load_image(evaldata_name)
	print("Succesfully loaded image")

	##load the model of interest and calculate predictions
	labels_cnn = oct.label_on_fly(pic,model_name,cwd_model,stride=stride,box_size = box_size)
	print("Succesfully calculated predictions")

	##create an output image of the raw predictions
	pic_prediction_cnn = oct.make_image(labels_cnn)
	print("Succesfully created raw prediction image")

	## Save the raw prediction
	oct.save_image(pic_prediction_cnn, prediction_cnn_name)
	print("Succesfully saved raw prediction image")

	##create labels of the predictions plus EM
	labels_cnn_em = oct.apply_em(labels_cnn)
	print("Succesfully calculated prediction using EM")

	##create an output image of the predictions plus EM
	pic_predictiom_cnn_em = oct.make_image(labels_cnn_em)
	print("Succesfully created final prediction image")

	## Save the raw prediction
	oct.save_image(pic_predictiom_cnn_em, prediction_cnn_name)
	print("Succesfully saved final prediction image")

	##Load the ground truth data
	labels_gt = oct.get_labels_gt(evaldata_gt_name)

	##Save the F1 scores in a txt file
	oct.create_f1_score(labels_gt,labels_cnn,labels_cnn_em)
