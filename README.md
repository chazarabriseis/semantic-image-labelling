########################################################################################## 
####    Code to train and  a CNN for semantic image pixel labelling of OCT images
####    Author: Julia Baldauf
####    5.8.2016	

required packages: gco_python (http://peekaboo-vision.blogspot.com.au/2012/05/graphcuts-for-python-pygco.html)

Set directories and box_size in Trainingflow and Evaluationflow

Trainingflwo firstly prepares the data by createing labels and cropped images of box_sizeXbox_size

It then trains a cnn inspired by Paisitkriangkrai et al.
(http://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W13/papers/Paisitkriangkrai_Effective_Semantic_Pixel_2015_CVPR_paper.pdf)

Evaluationsflow returns a labelled image with and without applying energy minimization 