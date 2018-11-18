
import sys

from packages.classifier import training

datadir = './preprocessed_img'
modeldir = './models/20180408-102900'
classifier_filename = './class/classifier.pkl'
eval_score_path = './class/accuracy_score.txt'
print ("Training Start")
obj = training(datadir, modeldir, classifier_filename, eval_score_path)
get_file = obj.main_train()
print('Saved classifier model to file "%s"' % get_file)
sys.exit("All Done")
