import json
import pickle
import numpy
from keras.datasets import cifar100

def createTree(file='meta'):
	ftr = open(file,'rb')
	labels = pickle.load(ftr,encoding='bytes')
	output = {}
	batches_x = {'root':[]}
	batches_y = {'root':[]}
	(x1,y_train_coarse),(a,b) = cifar100.load_data(label_mode='coarse')
	(x2,y_train_fine),(a,b) = cifar100.load_data(label_mode='fine')

	for i in range(y_train_fine.size):
		coarse_label = labels[b'coarse_label_names'][y_train_coarse[i][0]].decode('utf-8')
		fine_label = labels[b'fine_label_names'][y_train_fine[i][0]].decode('utf-8')
		if coarse_label not in output.keys():
			output[coarse_label] = {'fine':{},'val':str(y_train_coarse[i][0])}
		if fine_label not in output[coarse_label]['fine'].keys():
			output[coarse_label]['fine'][fine_label] = str(y_train_fine[i][0])
		if coarse_label not in batches_x.keys():
			batches_x[coarse_label] = []
			batches_y[coarse_label] = []
		batches_x[coarse_label] += x2[i]
		batches_y[coarse_label] += [y_train_fine[i]]
		batches_x['root'] += [x1[i]]
		batches_y['root'] += [y_train_coarse[i]]
	

	return output,batches_x,batches_y
	#import pdb
	#pdb.set_trace()
	#with open('cifar100_v2.json','w') as writefile:
	#	writefile.write(json.dumps(output))



if __name__ == "__main__" :
	createTree()

