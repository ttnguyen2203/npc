
import numpy as np 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random
import pandas as pd
from sklearn import svm 
from sklearn import metrics
from scipy import io


from preprocess import preprocessing
from preprocess import max_normalize
from preprocess import weight_mask
from preprocess import mean_mask
from preprocess import std_mask
from preprocess import extract_features


from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

from sklearn.preprocessing import MinMaxScaler
random.seed(1)
np.random.seed(1)

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
num_epochs = 10000
holdout = 0.2 #use k fold cross validation later
C = 10


model = svm.LinearSVC(C=C)
model2 = svm.SVC(kernel='poly', C=C, degree=2   , gamma='scale', coef0=1.0)



### TODO: REFACTOR crossValidation 

### 5-Fold Cross Validation ###
def crossValidation(C, trainX, trainY):
    #print(len(trainX))
    model = svm.LinearSVC(C=C, max_iter=1000)
    zipp = list(zip(trainX, trainY))
    random.shuffle(zipp)
    section_index = int(len(trainX) / 5)
    acc = []
    for i in range(5):
        if i == 4:
            valX, valY = zip(*zipp[i*section_index:])
            trainX, trainY = zip(*(zipp[0:i*section_index]))
        else:
            valX, valY = zip(*zipp[i*section_index:(i+1)*section_index])
            trainX, trainY = zip(*(zipp[0:i*section_index]+zipp[(i+1)*section_index:]))
        model.fit(trainX, trainY)
        acc.append(metrics.accuracy_score(valY, model.predict(valX), normalize=True, sample_weight=None))
    return sum(acc)/len(acc)




#################### MAIN ########################


cells = ['06B', '08D', '10A', '12B', '17B', '19B', '20A', '21A', '22A', '23A', '25C']

cellid   = '06B'
fc = 0.10 ## 0.15 to 0.05 is a good range for threshold

mat = io.loadmat('D:/Projects/NPC/v1_nvm_data/r02' + cellid + '_data.mat')
stim = timeFirst(mat['stim'])
stim, resp = removeNan(stim, mat['resp'])
#stim = np.reshape(stim, [stim.shape[0], 256])

features_selected = [48]    
#features_selected = [18, 44, 23, 81,  48, 97, 38]
scaler = MinMaxScaler()

features = extract_features(stim, resp, fc) 
features = np.transpose(features, [1,0])
features = np.array([features[i] for i in features_selected])
features = np.transpose(features, [1,0])
features = scaler.fit_transform(features)
#features = features / np.max(features)
print('ft', features.shape)

#stim = max_normalize(stim)
resp = np.reshape(resp, [stim.shape[0]])

frames, counts = preprocessing(stim, resp, fc)
#frames = max_normalize(frames)
#frames = np.reshape(frames, [frames.shape[0], 256])


data = frames
mask = weight_mask(frames, counts)
mask_arr = np.repeat(np.array([mask]), data.shape[0], axis=0)


#data -= mask_arr
data -= mean_mask(data)
data /= std_mask(data)
data = max_normalize(data)
data = np.reshape(data, [data.shape[0], 256])


print(features)


X_train, X_test, y_train, y_test = train_test_split(features,
    counts, train_size=0.75, test_size=0.25)



########################### CLASSIFIER ################################

# # validation_x = data[:int(holdout * len(frames))]
# # validation_y = labels[:int(holdout * len(frames))]
# # training_x = data[int(holdout * len(frames)):]
# # training_y = labels[int(holdout * len(frames)):]

# # model2.fit(training_x, training_y)
# # print(metrics.accuracy_score(validation_y, model2.predict(validation_x), normalize=True, sample_weight=None))

tpot = TPOTClassifier(generations=5, population_size=100, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))



################## UNIVARIATE FEATURE SELECTION #######################

# #apply SelectKBest class to extract top 10 best features
# bestfeatures = SelectKBest(score_func=f_classif, k='all')
# fit = bestfeatures.fit(features, counts)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame([i for i in range(256)])
# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(100,'Score'))  #print 10 best features



