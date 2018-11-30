import json
import pickle
import numpy
from keras.datasets import cifar100
from keras.utils import to_categorical
import pdb

def createTree(file='meta'):
	ftr = open(file,'rb')
	labels = pickle.load(ftr,encoding='bytes')
	output = {}
	batches_x = {'root':[]}
	batches_y = {'root':[]}

	val_batches_x = {'root':[]}
	val_batches_y = {'root':[]}

	(x1,y_train_coarse),_ = cifar100.load_data(label_mode='coarse')
	(x2,y_train_fine),_ = cifar100.load_data(label_mode='fine')

	x1 = x1/255
	x2 = x2/255
	x1_val = x1[::10]
	x2_val = x2[::10]
	y_val_fine = y_train_fine[::10]
	y_val_coarse = y_train_coarse[::10]
	x1 = [x1[i] for i in range(len(x1)) if i not in range(len(x1))[::10]]
	x2 = [x2[i] for i in range(len(x2)) if i not in range(len(x2))[::10]]
	y_train_fine = [y_train_fine[i] for i in range(len(y_train_fine)) if i not in range(len(y_train_fine))[::10]]
	y_train_coarse = [y_train_coarse[i] for i in range(len(y_train_coarse)) if i not in range(len(y_train_coarse))[::10]]

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
		batches_x[coarse_label] += [x2[i]]
		batches_y[coarse_label] += [y_train_fine[i]]
		batches_x['root'] += [x1[i]]
		batches_y['root'] += [y_train_coarse[i]]


	for i in range(y_val_fine.size):
		coarse_label = labels[b'coarse_label_names'][y_val_coarse[i][0]].decode('utf-8')
		fine_label = labels[b'fine_label_names'][y_val_fine[i][0]].decode('utf-8')

		if coarse_label not in val_batches_x.keys():
			val_batches_x[coarse_label] = []
			val_batches_y[coarse_label] = []

		val_batches_x[coarse_label] += [x2_val[i]]
		val_batches_y[coarse_label] += y_val_fine[i].tolist()
		val_batches_x['root'] += [x1_val[i]]
		val_batches_y['root'] += y_val_coarse[i].tolist()

	batches_x['root'] = numpy.array(batches_x['root'])
	val_batches_x['root'] = numpy.array(val_batches_x['root'])
	val_batches_y['root'] = numpy.array(val_batches_y['root'])

	for key in output.keys():
		batches_x[key] = numpy.array(batches_x[key])
		val_batches_x[key] = numpy.array(val_batches_x[key])
		val_batches_y[key] = numpy.array(val_batches_y[key])
	
	return output,batches_x,batches_y,val_batches_x,val_batches_y
	#import pdb
	#pdb.set_trace()
	#with open('cifar100_v2.json','w') as writefile:
	#	writefile.write(json.dumps(output))



if __name__ == "__main__" :
	createTree()

