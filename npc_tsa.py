### Time series analysis test

##Source code: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from math import sqrt
import random
import numpy
import numpy as np

import scipy.io
import pandas
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

"""
	Time: 
		-23 time steps at top of frame
		- 3 frames to change --> find way to auto identify these marks
		30, 31, 62, 63, 89 90

	TODO: 
	- Process input by eliminating outliers
		- Two passes: first pass eliminate dips during still frame, second eliminate transition frames

"""


def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
	model.add(Dropout(0.2))

	model.add(LSTM(units=neurons, return_sequences=True))  
	model.add(Dropout(0.2))

	model.add(LSTM(units=neurons))  
	model.add(Dropout(0.2)) 	
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

def forecast(model, batch_size, row):
	X = row[0:-1]
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
 
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def fixNan(series):
	nan_places = []
	for i in range(len(series)):
	    if np.isnan(series[i][0]):
	        nan_places.append(i)

	series = np.delete(series, nan_places, 0)
	return series

def toSupervised(series):
	
	# transform data to be stationary
	series = difference(series, 1)

	df = DataFrame()
	df['t'] = [series[i][0] for i in range(len(series))]
	df1 = concat([df.shift(-1)])
	df['start'] = [1 if 2 <= i % 28 <= 6 else 0 for i in range(len(series))]
	df['end'] = [1 if 1 <= i % 28 <= 5 else 0 for i in range(len(series))]
	df = concat([df,df1], axis=1)

	df.columns = ['t','start', 'end','t+1']
	# df['response'] = [series[i][0] for i in range(len(series))]

	#df = concat([df.shift(1), df], axis=1)
	df = df.fillna(0)
	print(df.head(100))
	df = df.values

	return df


np.random.seed(7)
mat = scipy.io.loadmat('D:/Projects/NPC/v1_nvmdata_full/v1_nvmdata_full/v1_nvm_data/r0206B_data.mat')
raw_values = fixNan(mat['resp'])


### TRAINING ###
df = toSupervised(raw_values)
train_size = int(len(df) * 0.5)
test_size = len(df) - train_size


test_set = df[train_size: len(df)]
train_set = df[0: train_size]

scaler, train_scaled, test_scaled = scale(train_set, test_set)

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 3, 10)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0:-1].reshape(len(train_scaled), 1, train_scaled.shape[1]-1)
lstm_model.predict(train_reshaped, batch_size=1)


# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train_set) + i]
	print('t=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[train_size:len(df)], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(raw_values[train_size:len(df)])
plt.plot(predictions)
plt.show()
