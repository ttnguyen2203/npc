
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


"""


    -TODO:
        - Fill in NaN values, status = Done (removing all NaN)
        - Tweak RNN skeleton, status = Paused
        - Data preprocessing, status = In Progress
        - Additional data extraction from stim, status = Not started
        - Treat the dataset as independent, classification problem, status = Not started


    -NOTES:
        -Problem Structure:
            -16x16xT stimulus input, Tx1 response outputs
            -response changes on order of ms (every timestep), each pixel changes every ~26 timesteps
                -makes sense for 23fps movie -> 26 X 16ms per frame change
            -stim has noise between each frame jump --> normalize
        - Fixing NaN
            - Suggested using regression or classification
            - Data set is a time series: 
                - Given data most likely: no trend/seasonality or only seasonality
                - Try: Multiple Imputation, XGBoost, Random Forest, KNN

                    - KNN: Given a (test) vector or image, find  k  vectors or images in Train Set that are 
                        "closest" to the (test) vector or image. k  closest vectors or images = k  labels. 
                        Assign the most frequent label to the (test) vector or image.

                        **COMMENT: data missing in continuous batches size ~ 25-40, KNN goes off of 
                        most frequent label, high risk of all data points in a batched assigned the same
                        value -> bad

            - Removing data points with NaN
        - Activation functions:
            - Keras:
                -ReLU
                -sigmoid/hardsigmoid
                -softmax
                -binary step 
                -linear
        
      
"""

### PREPROCESSING FUNCTIONS

#   Remove all nan entries
#   params: in: stim and resp from .mat file
#        out: stim and resp with nan entries removed

###DEBUG THESE 3 PASS BY VALUE
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


####
#   Normalize pixel input from range [0,255] to [-1,1]
def normalizePixels(stim):
    for i in range(len(stim)):
        stim[i] -= pixel_range / 2
        stim[i] /= pixel_range
    return stim


### PARAMS ###
num_epochs = 100
batch_size = 26
holdout = 8 #use k fold cross validation later


### MODELS ###
def make_NN():
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(16, 16)),
    keras.layers.Dense(512, activation='tanh', bias_regularizer=keras.regularizers.l2(0.01), 
    kernel_regularizer=keras.regularizers.l2(0.01), activity_regularizer=keras.regularizers.l2(0.01) ), #256 softsign/sigmoid > relu6 > elu/selu > relu/crelu
    keras.layers.Dropout(0.25),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(16, activation='softmax')
    ])
    
    return model







### PREPPING ###
mat = scipy.io.loadmat('D:/Projects/NPC/v1_nvmdata_full/v1_nvmdata_full/v1_nvm_data/r0206B_data.mat')
resp,stim = mat['resp'], mat['stim']
print(stim.shape, resp.shape)


nan_places = []
for i in range(len(resp)):
    if np.isnan(resp[i][0]):
        nan_places.append(i)

stim = np.transpose(stim, (2, 0, 1))
resp = np.delete(resp, nan_places, 0)
stim = np.delete(stim, nan_places, 0)





hold_out_thresh = int(len(stim) // 10 * holdout)

test_stim = stim[hold_out_thresh:]
test_resp = resp[hold_out_thresh:]
train_stim = stim[0:hold_out_thresh]
train_resp = resp[0:hold_out_thresh]
print(test_stim.shape, test_resp.shape, train_stim.shape, train_resp.shape)


### TRAINING ###
model = make_NN()
model.compile(optimizer=
            #tf.train.GradientDescentOptimizer(0.02),
            #keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
            keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False),
            #keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0),
            loss='sparse_categorical_crossentropy',
            #loss=keras.losses.mean_squared_error,
            metrics=['accuracy'])

history = model.fit(train_stim, train_resp,batch_size= batch_size, epochs=num_epochs)

test_loss, test_acc = model.evaluate(test_stim, test_resp)
print(test_loss, test_acc)

pred = model.predict(test_stim, verbose=1)
print('p', pred.shape)

pred = np.argmax(pred, axis=1)
plt.plot(pred)
plt.plot(test_resp)#np.argmax(test_resp, axis=1))
plt.show()
#plt.plot(history.history['loss'])
print(pred.shape)

