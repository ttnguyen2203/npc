

import numpy as np 
import tensorflow as tf
import scipy.io
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

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


#### NORMALIZATION METHODS
#   Normalize pixel input from range [0,255] to [0,1]
def max_normalize(stim):
    stim = stim / 255
    return stim

### LABELING FRAMES
def difference_from_next(stim):
	shifted = np.roll(stim, -1, axis=0)[0:len(stim)-1]
	stim = stim[0:len(stim)-1]
	abs_diff = np.abs(stim - shifted)

	#last frame is all zeros
	lf = np.zeros((16, 16))
	abs_diff = np.append(abs_diff, np.array([lf]), axis=0)
	print("abs_diff", abs_diff.shape)


	#abs_diff = np.sum(abs_diff, axis=(1,2))
	return abs_diff


"""
	- group similar looking frames together by:
		- calculate the difference in brightness from the next frame, pixelwise
		- calculate a threshold below which a frame is considered the same
		- boolean mask each frame, true if less than 
"""
def label_frames(stim, fc):
	diff = difference_from_next(stim)
	threshold = fc * np.amax(diff)
	mask = np.full(stim.shape, threshold)
	less = np.greater_equal(diff, mask)
	less = 1 * less #convert bool matrix to int matrix
	less = np.sum(less, axis=(1, 2))
	plt.plot(less)
	plt.plot(timeLast(stim)[8][8])
	plt.show()


	labels = [0 for i in range(len(stim))]
	# for i in range(0, len(stim) - 1):
	# 	if i == 0:
	# 		continue
	# 	# if diff[i - 1] < threshold:
	# 	# 	labels[i] = labels[i - 1]
	# 	# else:
	# 	# 	labels[i] = labels[i - 1] + 1
	# 	if diff[i - 1] < threshold:
	# 		labels[i] = labels[i - 1]
	# 	else:
	# 		labels[i] = int(not bool(labels[i - 1]))
	return labels


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






### TESTING:

cellid	 = 'r0206B'
#fc = 0.3
fc = 0.28

mat = scipy.io.loadmat('D:/Projects/NPC/v1_nvm_data/' + cellid + '_data.mat')
stim_og = mat['stim']
#stim_og = max_normalize(stim_og)
stim = timeFirst(mat['stim'])
stim, resp = removeNan(stim, mat['resp'])
# stim = max_normalize(stim)

# diff = np.sort(difference_from_next(stim))
label_frames = label_frames(stim, fc)

# plt.plot(stim_og[8][8])
# plt.plot(label_frames)


plt.plot(difference_from_next(stim))
plt.show()





