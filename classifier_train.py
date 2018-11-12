from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from packages.classifier import training

datadir = './preprocessed_img'
modeldir = './models/20180408-102900'
classifier_filename = './class/classifier.pkl'
print ("Training Start")
obj=training(datadir,modeldir,classifier_filename)
get_file=obj.main_train()
print('Saved classifier model to file "%s"' % get_file)
sys.exit("All Done")
