import json
import pickle
import numpy
from keras.datasets import cifar100
from keras.utils import to_categorical
import pdb
from sklearn.model_selection import train_test_split

def createTree(file='meta'):
	ftr = open(file,'rb')
	labels = pickle.load(ftr,encoding='bytes')
	json_file = open('cifar100_v3.json')
	json_str = json_file.read()
	json_data = json.loads(json_str)
	getcoarser = {}
	back_search = {}
	for key in json_data.keys():
		back_search[key] = []
		for coarse in json_data[key]['coarse']:
			getcoarser[coarse] = key
			back_search[coarse] = [key]
			for fine in json_data[key]['coarse'][coarse]['fine']:
				back_search[fine] = [key,coarse] 

	output = {}
	batches_x = {'root':[]}
	batches_y = {'root':[]}

	val_batches_x = {'root':[]}
	val_batches_y = {'root':[]}

	(x1,y_train_coarse),_ = cifar100.load_data(label_mode='coarse')
	(x2,y_train_fine),_ = cifar100.load_data(label_mode='fine')

	x1 = x1/255
	x2 = x2/255
	x0 = x1

	x1_val = x1[::5]
	x2_val = x2[::5]
	x0_val = x1_val
	y_val_fine = y_train_fine[::5]
	y_val_coarse = y_train_coarse[::5]
	x1 = [x1[i] for i in range(len(x1)) if i not in range(len(x1))[::5]]
	x2 = [x2[i] for i in range(len(x2)) if i not in range(len(x2))[::5]]
	y_train_fine = [y_train_fine[i] for i in range(len(y_train_fine)) if i not in range(len(y_train_fine))[::5]]
	y_train_coarse = [y_train_coarse[i] for i in range(len(y_train_coarse)) if i not in range(len(y_train_coarse))[::5]]
	#pdb.set_trace()
	for i in range(len(y_train_fine)):
		
		coarse_label = labels[b'coarse_label_names'][y_train_coarse[i][0]].decode('utf-8')
		#pdb.set_trace()
		fine_label = labels[b'fine_label_names'][y_train_fine[i][0]].decode('utf-8')
		coarser_label = getcoarser[coarse_label]
		if coarse_label not in output.keys():
			output[coarse_label] = {'fine':{},'val':str(y_train_coarse[i][0])}
		if fine_label not in output[coarse_label]['fine'].keys():
			output[coarse_label]['fine'][fine_label] = str(y_train_fine[i][0])
		if coarse_label not in batches_x.keys():
			batches_x[coarse_label] = []
			batches_y[coarse_label] = []
		if coarser_label not in  batches_x.keys():
			batches_x[coarser_label] = []
			batches_y[coarser_label] = []
		#pdb.set_trace()

		batches_x[coarse_label] += [x2[i]]
		batches_y[coarse_label] += [y_train_fine[i]]
		batches_x['root'] += [x0[i]]
		batches_y['root'] += [json_data[coarser_label]['val']]
		batches_x[coarser_label] += [x1[i]]
		batches_y[coarser_label] += [y_train_coarse[i]]


	for i in range(len(y_val_fine)):
		coarse_label = labels[b'coarse_label_names'][y_val_coarse[i][0]].decode('utf-8')
		fine_label = labels[b'fine_label_names'][y_val_fine[i][0]].decode('utf-8')
		coarser_label = getcoarser[coarse_label]
		if coarse_label not in val_batches_x.keys():
			val_batches_x[coarse_label] = []
			val_batches_y[coarse_label] = []
		if coarser_label not in val_batches_x.keys():
			val_batches_x[coarser_label] = []
			val_batches_y[coarser_label] = []

		val_batches_x[coarse_label] += [x2_val[i]]
		val_batches_y[coarse_label] += y_val_fine[i].tolist()
		val_batches_x['root'] += [x0_val[i]]
		val_batches_y['root'] += [json_data[coarser_label]['val']]#y_val_coarse[i].tolist()
		val_batches_x[coarser_label] += [x1[i]]
		val_batches_y[coarser_label] += [y_val_coarse[i]]

	# batches_x['root'] = numpy.array(batches_x['root'])
	# val_batches_x['root'] = numpy.array(val_batches_x['root'])
	# val_batches_y['root'] = numpy.array(val_batches_y['root'])

	for key in batches_x.keys():
		batches_x[key] = numpy.array(batches_x[key])
		val_batches_x[key] = numpy.array(val_batches_x[key])
		val_batches_y[key] = numpy.array(val_batches_y[key])
	
	return json_data,batches_x,batches_y,val_batches_x,val_batches_y, back_search, 
		[label[b'fine_label_names'][i].decode('utf-8') for i in range(label[b'fine_label_names'])]
	#import pdb
	#pdb.set_trace()
	#with open('cifar100_v2.json','w') as writefile:
	#	writefile.write(json.dumps(output))



if __name__ == "__main__" :
	createTree()

