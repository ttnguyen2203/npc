import numpy as np 
import tensorflow as tf
import scipy.io
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.ndimage import generic_filter
from tensorflow import keras
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import keras.backend as K


# from keras.preprocessing import image
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input

"""
MODELS USED:


DESIGN CORRECTION:
 - Find a way to eliminate NaN's effect --> current = removing all entries with NaN in them
 - Data not uniform: design a way to iterate through sections of data to use_bias

"""
seed = 1
#np.random.seed(seed)
pixel_range = 255 #pixel input range



def cutoff(data_length, batch_size):
	return data_length - (data_length % batch_size)

#Preprocessing functions:
def normalizePixels(stim):
	for i in range(len(stim)):
		stim[i] -= pixel_range / 2
		stim[i] /= pixel_range
	return stim


def naivePreprocess(stim):
	return stim

def averageMask(stim):
	for i in range(len(stim)):
		stim[i] -= np.mean(stim[i])
	return stim

def custom_roll(a, axis=0):
	n = 3
	a = a.T if axis==1 else a

	pad = np.zeros((n-1, a.shape[1]))
	a = np.concatenate([a, pad], axis=0)
	ad = np.dstack([np.roll(a, i, axis=0) for i in range(n)])
	a = ad.sum(2)[1:-1, :]

	a = a.T if axis==1 else a
	return a

def surrounding_sum(stim):
	new_arr = []
	for A in stim:
		new_arr.append(custom_roll(custom_roll(A), axis=1) - A)
	return np.stack(new_arr)

"""
assign to each pixel the avg of its 8 neighbors if applicable
"""


def avg_neighbor(stim):
	def eight_neighbor_average_convolve2d(x):
	    kernel = np.ones((3, 3))
	    kernel[1, 1] = 0

	    neighbor_sum = convolve2d(
	        x, kernel, mode='same',
	        boundary='fill', fillvalue=0)

	    num_neighbor = convolve2d(
	        np.ones(x.shape), kernel, mode='same',
	        boundary='fill', fillvalue=0)

	    return neighbor_sum / num_neighbor

	mask = np.ones((3, 3))
	mask[1, 1] = 0
	new_arr = []
	for A in stim:
		new_arr.append(eight_neighbor_average_convolve2d(A))
	return new_arr


"""
	NOTES:
		
"""

cellID = ['r0206B', '']

mat = scipy.io.loadmat('D:/Projects/NPC/v1_nvmdata_full/v1_nvmdata_full/v1_nvm_data/r0206B_data.mat')

resp,stim = mat['resp'], mat['stim']


#visualizing missing data
#resp_pandas = pd.DataFrame(resp)
#msno.matrix(resp_pandas)
#plt.show()
#plt.plot(resp_pandas)
#plt.show()


nan_places = []
#cleans resp of NaN
for i in range(len(resp)):
	if np.isnan(resp[i][0]):
		#resp[i] = 
		nan_places.append(i)
#print(nan_places)
		
# resp = np.multiply(np.divide(resp, 15), 255)

#analysis
plt.plot(stim[8][8])
plt.plot(resp)
plt.show()




#transposing array from (16 x 16 x T) to (T x 16 x 16)
stim = np.transpose(stim, (2, 0, 1))


resp = np.delete(resp, nan_places, 0)
stim = np.delete(stim, nan_places, 0)





# #average
# avg = averageMask(stim)

# #surrounding sums
# ss = surrounding_sum(stim)

# #neighbors avg
# navg = avg_neighbor(stim)

#normalize
stim = normalizePixels(stim)

#combining processes
combination = [stim]
num_feature = len(combination)*16*16
stim=np.stack(combination, 3)


# #Reshape for RNN
# stim = np.reshape(stim, (len(stim), num_feature))


hold_out_thresh = int(len(stim) // 10 * 8)

test_stim = stim[hold_out_thresh:]
test_resp = resp[hold_out_thresh:]
stim = stim[0:hold_out_thresh]
resp = resp[0:hold_out_thresh]

print(stim.shape)

# batch_size = 5

# #make batches for RNN
# cutoff = cutoff(stim.shape[0], batch_size)
# stim = stim[0:cutoff]
# resp = resp[0:cutoff]

# stim = np.stack(np.split(stim, stim.shape[0] // batch_size))
# resp = np.stack(np.split(resp, resp.shape[0] // batch_size))


# #stim = np.reshape(stim, (1, stim.shape[0],stim.shape[1]))
# #resp = np.reshape(resp, (1, resp.shape[0], 1))

# test_stim = np.reshape(test_stim, (test_stim.shape[0], 1, test_stim.shape[1]))
# test_resp = np.reshape(test_resp, (test_resp.shape[0], 1))


#print(test_resp.shape)
#print(len(stim))		


#Training 

def make_NN():
	model = keras.Sequential([
	#keras.layers.Flatten(input_shape=(16, 16)),
	keras.layers.Dense(512, activation='softplus', bias_regularizer=keras.regularizers.l2(0.01), 
	kernel_regularizer=keras.regularizers.l2(0.01), activity_regularizer=keras.regularizers.l2(0.01) ), #256 softsign/sigmoid > relu6 > elu/selu > relu/crelu
	keras.layers.Dropout(0.25),
	keras.layers.Dense(512, activation='relu'),
	keras.layers.Dense(512, activation='relu'),
	keras.layers.Dropout(0.3),

	# keras.layers.Dense(300, activation='relu'),
	# keras.layers.Dropout(0.5),
	# keras.layers.Dense(200, activation='relu'),
	# #keras.layers.Dense(15, activation='softplus') 
	keras.layers.Dense(16, activation='softmax')
	])
	
	return model

def make_RNN():
	model = keras.Sequential()

	model.add(keras.layers.LSTM(num_feature, activation='tanh', recurrent_activation='hard_sigmoid', 
		use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
		bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, 
		bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, 
		bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, 
		return_state=False, go_backwards=False, stateful=False, unroll=False))

	# model.add(keras.layers.SimpleRNN(num_feature, activation='softplus', use_bias=True, kernel_initializer='glorot_uniform', 
	# 	recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, 
	# 	bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
	# 	dropout=0.0, recurrent_dropout=0.0, return_sequences=True, return_state=False, go_backwards=True, stateful=False, unroll=False))


	# Fully connected layer
	#model.add(keras.layers.Dense(100, activation='relu'))
	model.add(keras.layers.Dense(100, activation='softplus'))
	#model.add(keras.layers.LeakyReLU(alpha=0.3))
	#Dropout for regularization
	#model.add(keras.layers.Dropout(0.3))

	# Output layer
	model.add(keras.layers.Dense(15, activation='softsign'))
	model.add(keras.layers.Dense(16, activation='softmax'))

	return model


#reshaping matrices for cnn

# stim = stim.reshape(stim.shape[0], 16, 16, 1)
# test_stim = test_stim.reshape(test_stim.shape[0], 16, 16, 1)
# resp = keras.utils.to_categorical(resp, 16)
# test_resp = keras.utils.to_categorical(test_resp, 16)

def make_CNN():
	model = keras.models.Sequential()
	model.add(keras.layers.Conv2D(64, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=(16,16,1)))
	model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Dropout(0.25))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(200, activation='relu'))
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Dense(16, activation='softmax'))
	return model

model = make_NN()

#custom loss function
def squared_error(true, predicted):
	diff = (predicted - true)**2
	return np.sum(diff)

def mean_rounding_loss(ytrue,ypred): #has derivative, that is equal to the nearest positive natural number
  x=ytrue-ypred
  a = K.round(K.abs(x))
  return K.mean(a*(a-1)/4+a*x,axis=-1)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

model.compile(optimizer=#tf.train.GradientDescentOptimizer(0.02),
			#keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
			keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False),
			#keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0),
			#loss=keras.losses.mean_squared_error, 
			loss='mae',
			metrics=['accuracy'])

# model.compile(loss='mae',
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])


history = model.fit(stim, resp,epochs=10)


print(test_stim.shape)
print(test_resp.shape)
test_loss, test_acc = model.evaluate(test_stim, test_resp)
print(test_loss, test_acc)

pred = model.predict(test_stim, verbose=1)
print('p', pred.shape)

pred = np.argmax(pred, axis=1)
print(pred.shape)
#pred = np.reshape(pred, (pred.shape[0], 1))

# plt.subplot(2, 1, 1)
# plt.plot(pred, 'o-')
# plt.title('A tale of 2 subplots')
# plt.ylabel('Prediction')

# plt.subplot(2, 1, 2)
# plt.plot(test_resp, '.-')
# plt.xlabel('time (s)')
# plt.ylabel('test_resp')


plt.plot(pred)
plt.plot(np.argmax(test_resp, axis=1))

#plt.plot(history.history['loss'])
plt.show()


