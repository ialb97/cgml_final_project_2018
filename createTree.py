import json
import pickle
import numpy
from keras.datasets import cifar100

def createTree(file='meta'):
	ftr = open(file,'rb')
	labels = pickle.load(ftr,encoding='bytes')
	output = {}
	(x1,y_train_coarse),(a,b) = cifar100.load_data(label_mode='coarse')
	(x2,y_train_fine),(a,b) = cifar100.load_data(label_mode='fine')
	for i in range(y_train_fine.size):
		coarse_label = labels[b'coarse_labels'][y_train_coarse[i]]
		fine_label = labels[b'fine_labels'][y_train_fine[i]]
		if course_label not in output.keys():
			output[course_label] = {'fine':{},'val':y_train_coarse[i]}
		if fine_label not in output[course_label]['fine'].keys():
			output[coarse_label]['fine'][fine_label] = y_train_fine[i]
	with open('cifar100_v2.json','w') as writefile:
		writefile.write(json.dump(output))


if __name__ == "__main__" :
	createTree()

