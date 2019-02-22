"""
RNN Skeleton code by Erik Hallstrom
https://medium.com/@erikhallstrm/using-the-tensorflow-lstm-api-3-7-5f2b97ca6b73

"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


"""


    -TODO:
        - Fill in NaN values, status = Researching
        - Tweak RNN skeleton, status = Not started
        - Sophisticated data preprocessing, status = Not started
        - Additional data extraction from stim, status = Not started
        - Treat the dataset as independent, classification problem, status = Not started


    -NOTES:
        - Observation from running classification model: (not concrete, classification model needs more refining)
            - Data is not regular:
                - cell r0206B learns better if train on the latter 50% of data set rather than first
            - Cell 0210A has 95% accuracy with clasification model with the following hyperparams:
                - Preprocessing: Average Masks, remove NaN data points
                - 1:1 training to hold out, training on second half
                - Hidden layers: 256 softsign, 400 sigmoid, 15 softmax
                - Gradient Descent Optimizer lr = 0.005
                - loss = sparse categorical crossentropy
                - metric = accuracy

                **COMMENT: acc plateaus at 95% at ~epoch 2 and stays the same throughout, weird behavior.





        - Fixing NaN
            - Suggested using regression or classification
            - Data set is a time series: 
                - Linear Regression model: Yt = mt + st + et (trend, seasonality, random variable)
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
                -






        
"""


mat = scipy.io.loadmat('D:/Projects/NPC/v1_nvmdata_full/v1_nvmdata_full/v1_nvm_data/r0206B_data.mat')
resp,stim = mat['resp'], mat['stim']


###PARAMS

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length
