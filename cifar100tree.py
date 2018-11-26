from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.utils import to_categorical
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import createTree
import random
import pdb


class cifar100tree:
	def __init__(self,weights=None,learning_rate=.001):
		self.batch_size = 32
		self.num_classes = 100
		self.weight_decay = 0.0005
		self.x_shape = [32,32,3]

		self.inputs, self.base_model = self.build_base_model()
		self.vgg_model = self.build_vgg_model(self.inputs,self.base_model)
		if (weights):
			self.vgg_model.load_weights(weights)

		self.tree,self.x_batches,self.y_batches,self.val_x_batches,self.val_y_batches = createTree.createTree()
		self.get_root_mapping()
		self.y_batches,self.mapping,self.reverse_mapping = self.one_hot(self.y_batches)
		
		self.learning_rate = learning_rate
		self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)
		self.model_dict = self.build_model_dict(self.base_model,self.inputs)
		# import pdb
		# pdb.set_trace()
		self.fit()
		
	def build_base_model(self):
		inp = Input(shape=self.x_shape)

		weight_decay = self.weight_decay

		conv1_c = Conv2D(64, (3, 3), padding='same',
						 input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay))
		conv1_a = Activation('relu')
		conv1_b = BatchNormalization()
		conv1_d = Dropout(0.3)

		conv1 = conv1_d(conv1_b(conv1_a(conv1_c(inp))))

		conv2_c = Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))
		conv2_a = Activation('relu')
		conv2_b = BatchNormalization()
		conv2_p = MaxPooling2D(pool_size=(2, 2))

		conv2 = conv2_p(conv2_b(conv2_a(conv2_c(conv1))))

		conv3_c = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))
		conv3_a = Activation('relu')
		conv3_b = BatchNormalization()
		conv3_d = Dropout(0.4)

		conv3 = conv3_d(conv3_b(conv3_a(conv3_c(conv2))))

		conv4_c = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))
		conv4_a = Activation('relu')
		conv4_b = BatchNormalization()
		conv4_p = MaxPooling2D(pool_size=(2, 2))

		conv4 = conv4_p(conv4_b(conv4_a(conv4_c(conv3))))

		conv5_c = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))
		conv5_a = Activation('relu')
		conv5_b = BatchNormalization()
		conv5_d = Dropout(0.4)

		conv5 = conv5_d(conv5_b(conv5_a(conv5_c(conv4))))

		conv6_c = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))
		conv6_a = Activation('relu')
		conv6_b = BatchNormalization()
		conv6_d = Dropout(0.4)

		conv6 = conv6_d(conv6_b(conv6_a(conv6_c(conv5))))

		conv7_c = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))
		conv7_a = Activation('relu')
		conv7_b = BatchNormalization()
		conv7_p = MaxPooling2D(pool_size=(2, 2))

		conv7 = conv7_p(conv7_b(conv7_a(conv7_c(conv6))))

		conv8_c = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))
		conv8_a = Activation('relu')
		conv8_b = BatchNormalization()
		conv8_d = Dropout(0.4)

		conv8 = conv8_d(conv8_b(conv8_a(conv8_c(conv7))))

		conv9_c = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))
		conv9_a = Activation('relu')
		conv9_b = BatchNormalization()
		conv9_d = Dropout(0.4)

		conv9 = conv9_d(conv9_b(conv9_a(conv9_c(conv8))))

		conv10_c = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))
		conv10_a = Activation('relu')
		conv10_b = BatchNormalization()
		conv10_p = MaxPooling2D(pool_size=(2, 2))

		conv10 = conv10_p(conv10_b(conv10_a(conv10_c(conv9))))

		conv11_c = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))
		conv11_a = Activation('relu')
		conv11_b = BatchNormalization()
		conv11_d = Dropout(0.4)

		conv11 = conv11_d(conv11_b(conv11_a(conv11_c(conv10))))

		conv12_c = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))
		conv12_a = Activation('relu')
		conv12_b = BatchNormalization()
		conv12_d = Dropout(0.4)

		conv12 = conv12_d(conv12_b(conv12_a(conv12_c(conv11))))

		conv13_c = Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay))
		conv13_a = Activation('relu')
		conv13_b = BatchNormalization()
		conv13_p = MaxPooling2D(pool_size=(2, 2))
		conv13_d = Dropout(0.5)
		conv13_f = Flatten()

		conv13 = conv13_f(conv13_d(conv13_p(conv13_b(conv13_a(conv13_c(conv12))))))
		
		dense1_d = Dense(512,kernel_regularizer=regularizers.l2(weight_decay))
		dense1_a = Activation('relu')
		dense1_b = BatchNormalization()
		dense1_do = Dropout(0.5)

		dense1 = dense1_do(dense1_b(dense1_a(dense1_d(conv13))))

		return inp,dense1;

	def build_vgg_model(self,inp,base_model):
		dense2_d = Dense(self.num_classes)
		dense2_a = Activation('softmax')

		dense2 = dense2_a(dense2_d(base_model))

		model = Model(inputs=inp,outputs=dense2)
		
		return model

	def normalize(self,X_train,X_test):
		#this function normalize inputs for zero mean and unit variance
		# it is used when training a model.
		# Input: training set and test set
		# Output: normalized training set and test set according to the trianing set statistics.
		mean = np.mean(X_train,axis=(0,1,2,3))
		std = np.std(X_train, axis=(0, 1, 2, 3))
		print(mean)
		print(std)
		X_train = (X_train-mean)/(std+1e-7)
		X_test = (X_test-mean)/(std+1e-7)
		return X_train, X_test

	def normalize_production(self,x):
		#this function is used to normalize instances in production according to saved training set statistics
		# Input: X - a training set
		# Output X - a normalized training set according to normalization constants.

		#these values produced during first training and are general for the standard cifar10 training set normalization
		mean = 121.936
		std = 68.389
		return (x-mean)/(std+1e-7)

	def predict(self,x,normalize=True,batch_size=50):
		if normalize:
			x = self.normalize_production(x)
		return self.model.predict(x,batch_size)


	def build_model(self,base_model,inputs,outputs):
		dense2_d = Dense(outputs)
		dense2_a = Activation('softmax')

		dense2 = dense2_a(dense2_d(base_model))
		return Model(inputs=inputs,outputs=dense2)

	def build_model_dict(self,base_model,inputs):
		models = {}

		models['root'] = self.build_model(self.base_model,inputs,len(self.tree))
		models['root'].compile(self.optimizer, 
								metrics=['accuracy'],
								loss='categorical_crossentropy')

		for key in self.tree:
			outputs = len(self.tree[key]['fine'])
			models[key] = self.build_model(self.base_model,inputs,outputs)
			models[key].compile(self.optimizer, 
								metrics=['accuracy'],
								loss='categorical_crossentropy')
		return models

	def one_hot(self,labels):
		mapping = {}
		reverse_mapping = {}
		new_batches = {}
		for key in labels:
			new_batches[key] = []
			mapping[key] = {}
			reverse_mapping[key] = []
			i=0
			for entry in labels[key]:
				val = entry[0]
				if (val not in mapping[key].keys()):
					mapping[key][val]=i
					reverse_mapping[key] += [val]
					i+=1
				new_batches[key] += [mapping[key][val]]
			new_batches[key] = to_categorical(new_batches[key],i)
		return new_batches, mapping, reverse_mapping

	def fit(self):
		batch_iters = {}
		datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
		k = 0
		keys = ['root']
		batches = [datagen.flow(self.x_batches['root'],self.y_batches['root'],batch_size=self.batch_size)]
		num_batches = len(batches[k])
		batches_per = [len(batches[k])]
		k += 1
		
		for key in self.tree:
			keys += [key]
			batches += [datagen.flow(self.x_batches[key],self.y_batches[key],batch_size=self.batch_size)]
			num_batches += len(batches[k])
			batches_per += [len(batches[k])]
			k += 1

		# pdb.set_trace()
		
		for i in range(num_batches):
			rng = random.randint(0,len(self.tree)+1)
			if batches_per[rng]:
				batches_per[rng] -= 1
				x_batch,y_batch = batches[rng][batches_per[rng]]
				self.model_dict[keys[rng]].train_on_batch(x_batch,y_batch)
				print("Batch:{:3.0f}/{}".format(i,num_batches),end='\r')
			else:
				i -= 1

		# pdb.set_trace()
		for model in model_dict:
			model_dict[model].save_weights('weights/cifar100tree_{}.h5'.format(model))
		print("accuracy: {}".format(self.eval_on_batch(datagen.flow(self.val_x_batches,self.val_y_batches,batch_size=1))))

	def get_root_mapping(self):
		self.root_mapping = [0]*len(self.tree)
		for key in self.tree:
			self.root_mapping[int(self.tree[key]['val'])] = key

	def eval(self,x,y):
		result = self.model_dict['root'].predict_on_batch(x)
		val = np.where(result)[0][0]
		key = self.root_mapping[val]
		
		result = self.model_dict[key].predict_on_batch(x)
		val = np.where(result)[0][0]
		key = self.reverse_mapping[key][val]
		one_hot = to_categorical(key,self.num_classes)

		return one_hot == y

	def eval_on_batch(self,x_batch,y_batch):
		# returns accuracy metric for batch
		# note: currently batch size needs to be one
		# TODO: make it not need to be 1
		correct = 0
		for x in x_batch:
			if self.eval(x,y):
				correct += 1
		return correct/len(x_batch)





if __name__ == '__main__':
	(x_train, y_train), (x_test, y_test) = cifar100.load_data()
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	y_train = keras.utils.to_categorical(y_train, 100)
	y_test = keras.utils.to_categorical(y_test, 100)

	model = cifar100tree(weights="weights/cifar100vgg.h5")

	# predicted_x = model.predict(x_test)
	# residuals = (np.argmax(predicted_x,1)!=np.argmax(y_test,1))
	# loss = sum(residuals)/len(residuals)
	# print("the validation 0/1 loss is: ",loss)