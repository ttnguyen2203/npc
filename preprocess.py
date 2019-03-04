

import numpy as np 
import tensorflow as tf
import scipy.io
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""
- FC: Frame Cutoff: the cut off percentage of summed difference between two consecutive frames
		 below which those frames are considered to be the same
	- 
"""



#	Remove all nan entries
#   params: in: stim and resp from .mat file
#        out: stim and resp with nan entries removed

def removeNan(stim, resp):
    nan_places = []
    for i in range(len(resp)):
        if np.isnan(resp[i][0]):
            nan_places.append(i)
    return np.delete(stim, nan_places, 0), np.delete(resp, nan_places, 0)

#   Convert stim matrix to Tx16x16 or 16x16xT
def timeFirst(stim):
    return np.transpose(stim, (2, 0, 1))

def timeLast(stim):
    return np.transpose(stim, (1, 2, 0))


########## NORMALIZATION METHODS ##########
#   Normalize pixel input from range [0,255] to [0,1]
def max_normalize(stim):
    stim = stim / 255
    return stim

########## LABELING FRAMES ##########
def difference_from_next(stim):
	shifted = np.roll(stim, -1, axis=0)[0:len(stim)-1]
	stim = stim[0:len(stim)-1]
	abs_diff = np.abs(stim - shifted)

	#last frame is all zeros
	lf = np.zeros((16, 16))
	print('abs_diff', abs_diff.shape)
	print(lf.shape)
	abs_diff = np.append(abs_diff, np.array([lf]), axis=0)

	#abs_diff = np.sum(abs_diff, axis=(1,2))
	return abs_diff


"""
	- group similar looking frames together by:
		- calculate the difference in brightness from the next frame, pixelwise
		- calculate a threshold below which a frame is considered the same
		- boolean mask each frame, true if above threshold
		- sum 0 and 1's, all frame with 0 sum are considered same
"""
def group_frames(stim, fc):
	diff = difference_from_next(stim)
	threshold = fc * np.amax(diff)
	mask = np.full(stim.shape, threshold)
	binary_mask = 1 * np.greater_equal(diff, mask)  #convert bool matrix to int matrix
	sum_vals = np.sum(binary_mask, axis=(1, 2))
	return sum_vals

### grouping counts by 10's
def group_counts(counts):
	return np.round(np.divide(counts, 10)) #rounding
	#return np.floor_divide(counts, 10) # floor division

def group_counts_binary_threshold(counts, threshold):
	mask = np.repeat(threshold, counts.shape[0])
	return np.greater_equal(counts, mask) * 1

# enter increasing threshold to label
def group_counts_multithreshold(counts, thresholds):
	levels = []
	for t in thresholds:
		mask = np.repeat(t, counts.shape[0])
		levels.append(np.greater_equal(counts, mask) * 1)

	return np.sum(np.array(levels), axis=0)


"""
	- averages pixel values of grouped frames
	- sum spike counts in corresponding time
"""
def get_frames(stim, resp, groups):
	# check dimensions of inputs
	if stim.shape[1] != 16 or stim.shape[2] != 16:
		stim = timeFirst(stim)

	frames = list()
	AP_count = list()
	i = 0

	#avg frames
	while i < len(stim):
		if groups[i] == 0:
			s = i
			while i < len(stim) and groups[i] == 0:
				i += 1
			e = i - 1
			mean = np.mean(stim[s:e], axis=0)
			frames.append(mean)
			AP_count.append(np.sum(resp[s:e]))
		else:
			i += 1
	return np.array(frames), np.array(AP_count)


### binary thresholds:
#	30 - 82% acc

thresholds = [10, 20, 30, 40]

def preprocessing(stim, resp, fc):
	groups = group_frames(stim, fc)
	frames, counts = get_frames(stim, resp, groups)
	#counts = group_counts(counts)
	#counts = group_counts_binary_threshold(counts, 30)
	counts = group_counts_multithreshold(counts, thresholds)
	print("counts shape", counts.shape)
	return frames, counts

def cnn_preprocessing(stim, resp, fc):
	groups = group_frames(stim, fc)
	frames, counts = get_frames(stim, resp, groups)
	#counts = group_counts(counts)
	#counts = group_counts_binary_threshold(counts, 30)
	data = frames
	mask = weight_mask(frames, counts)
	mask_arr = np.repeat(np.array([mask]), data.shape[0], axis=0)


	#data -= mask_arr
	data -= mean_mask(data)
	data /= std_mask(data)
	data = max_normalize(data)
	counts = group_counts_multithreshold(counts, thresholds)
	return frames, counts


### deprecated
def label_by_pixel(stim, fc):
	pixelid = (8, 8)

	diff = difference_from_next(stim)
	threshold = fc * np.amax(diff)
	labels = [0 for i in range(len(stim))]
	diff_T = timeLast(diff)
	pixel = diff_T[pixelid[0]][pixelid[1]]

	for i in range(0, len(stim)):
		if i == 0:
			continue
		if pixel[i - 1] < threshold:
			labels[i] = labels[i - 1]
		else:
			labels[i] = int(not bool(labels[i - 1]))
	return labels





############### FEATURE EXTRACTION ###########################

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, MaxPool2D
 



def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
     
    return model

def lenet_5():
	model = Sequential()
	model.add(Conv2D(filters=20, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
	model.add(MaxPool2D())
	model.add(Conv2D(filters=50, kernel_size=5, padding='same', activation='relu'))
	model.add(MaxPool2D())
	model.add(Flatten())

	return model

def alexnet():
    model = Sequential()
    model.add(Conv2D(96, 11, strides=4, activation='relu'))
    model.add(MaxPool2D(3, 2))
    model.add(Conv2D(256, 5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(3, 2))
    model.add(Conv2D(384, 3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(3, 2))
    model.add(Flatten())
    return model



def extract_features(stim, resp, fc):
	data, counts = cnn_preprocessing(stim, resp, fc)
	data = np.reshape(data, [data.shape[0], 16, 16, 1])
	input_shape = data.shape[1:]

	model = createModel()
	batch_size = 100
	epochs = 100

	features = model.predict(data, batch_size, verbose=1)
	return features




#### classifier ########

from keras import layers
from keras.models import Model

def alexnet_classifier(in_shape, n_classes=5, opt='sgd'):
    model = Sequential()
    model.add(Conv2D(96, 11, strides=4, activation='relu'))
    model.add(MaxPool2D(3, 2))
    model.add(Conv2D(256, 5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(3, 2))
    model.add(Conv2D(384, 3, strides=1, padding='same', activation='relu'))
    model.add(Conv2D(256, 3, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(3, 2))
    model.add(Flatten())


    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
    return model

########## ANALYSIS ##########

"""
For each time point, take the associated frame and multiply pixel values by # AP's at that time point

so frames + AP --> weighted frames

then add all weighted frames

into one frame

and then divide all pixel values in that one frame by the total number of AP's over all time points

"""

def weight_mask(data, resp):

	resp = np.reshape(resp, (resp.shape[0], 1))
	data = np.reshape(data, (data.shape[0], 16*16))
	weighted = np.reshape(data * resp, (data.shape[0], 16, 16))
	mask = np.sum(weighted, axis=0)

	return mask

def mean_mask(data):
	return np.mean(data, axis=0)

def std_mask(data):
	return np.std(data, axis=0)





# ########## TESTING ##########

cells = ['06B', '08D', '10A', '12B', '17B', '19B', '20A', '21A', '22A', '23A', '25C']

cellid	 = '06B'
fc = 0.10 ## 0.15 to 0.05 is a good range for threshold

mat = scipy.io.loadmat('D:/Projects/NPC/v1_nvm_data/r02' + cellid + '_data.mat')
stim_og = mat['stim']
#stim_og = max_normalize(stim_og)
stim = timeFirst(mat['stim'])
stim, resp = removeNan(stim, mat['resp'])
# stim = max_normalize(stim)

# # diff = np.sort(difference_from_next(stim))
# groups = group_frames(stim, fc)
# frames, counts = get_frames(stim, resp, groups)

# plt.plot(np.sort(counts))
# plt.plot(np.sort(group_counts(counts)))
# plt.show()

# groups = group_frames(stim, fc)
# frames, counts = get_frames(stim, resp, groups)
# weight_mask(frames, counts)




### Extract feature

data, counts = cnn_preprocessing(stim, resp, fc)
data = np.reshape(data, [data.shape[0], 16, 16, 1])
input_shape = data.shape[1:]

model = createModel()
model2 = lenet_5()


classifier = alexnet_classifier(input_shape)

X_train, X_test, y_train, y_test = train_test_split(data,
    counts, train_size=0.75, test_size=0.25)

classifier.fit(X_train, y_train, epochs=100)
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)


# batch_size = 100
# epochs = 100

# features = model2.predict(data, batch_size, verbose=1)
# print(features.shape)

# model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
 
# history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, 
#                    validation_data=(test_data, test_labels_one_hot))
 
# model1.evaluate(test_data, test_labels_one_hot)





# ### plot summed AP spikes during frame, sorted
# for cellid in cells:
# 	mat = scipy.io.loadmat('D:/Projects/NPC/v1_nvm_data/r02' + cellid + '_data.mat')
# 	stim_og = mat['stim']
# 	#stim_og = max_normalize(stim_og)
# 	stim = timeFirst(mat['stim'])
# 	stim, resp = removeNan(stim, mat['resp'])
# 	groups = group_frames(stim, fc)
# 	frames, counts = get_frames(stim, resp, groups)
# 	plt.plot(np.sort(counts))
# 	plt.title('Summed APs during a frame, cell r02'+ cellid)
# 	plt.savefig('summedAP_r02'+cellid)
# 	plt.show()





