# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:13:01 2020
Music Genre Classifier based on music files spectral information.
Using Random Forest Classifiers and Tensorflow
@author: santi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

data = pd.read_csv('datasets\\data.csv')
tf.keras.backend.clear_session()
'''
Filenames in the data sheet are of the structure genre.number.au
Using the split method to properly obtain the labels for each track
'''

data['genre'] = data['filename'].str.split('.').str[0]

'''
Looking for heavily correlated variables to potentially drop.
Splitting the dataframe as X, y with the labels being the y set
'''

X = data.drop(['filename','genre'], axis = 1)
y = data['genre']

plt.figure(figsize = (16,9))
sns.heatmap(X.corr())

'''
Since tracks are of equal duration, it is logical that tempo (beats per min)
and beats are strongly correlated. Since all data follows this structure
the beats column will be dropped
'''

X.drop('beats', axis = 1, inplace = True)

'''
Spectral centroid, spectral bandwidth and rolloff are all metrics related
to track energy, rolloff being a percentile, bandwidth an interval, and
centroid a fixed value. Rolloff or bandwith are more clearly descriptive of
the tracks' energy, therefore, we're only keeping rolloff
'''

X.drop(['spectral_centroid','spectral_bandwidth'], axis = 1, inplace = True)

plt.figure(figsize = (16,9))
sns.heatmap(X.corr())



'''
Initial model, no preprocessing
'''

encodelab=LabelEncoder()
encodelab.fit(y)

X2 = X.values
y2 = encodelab.transform(y.values)
n_split=5
labels = list(y.unique())
for tr_id,ts_id in KFold(n_split,shuffle = True,random_state = 42).split(X2):
  x_train,x_test=X2[tr_id],X2[ts_id]
  y_train,y_test=y2[tr_id],y2[ts_id]
  

forest = RandomForestClassifier(n_estimators = 1000)
forest.fit(x_train,y_train)

preds = forest.predict(x_test)
print(classification_report(y_test, preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))

plt.figure(figsize = (16,9))
sns.heatmap(confusion_matrix(y_test, preds), xticklabels = labels,
            yticklabels = labels, annot = True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest on Unscaled Data')

'''
Random Forest Classifier seems to be very effective when classifying classical
music; however, this is expected for nearly every model since compression
is barely used in classical music and therefore it has different energy values

Analyzing feature importances
'''
imp = pd.DataFrame({'feature': X.columns,
                    'importance': forest.feature_importances_})
imp.columns = imp.columns.str.strip()
imp.sort_values(by=['importance'], ascending = False, inplace = True)

plt.figure(figsize=(16,9))
impchart = sns.barplot(x = 'feature',y = 'importance', palette = 'plasma', 
                       data = imp)
impchart.set_xticklabels(impchart.get_xticklabels(), rotation = 45)
plt.title('Variable importance')

'''
It is seen that the chromagram (related to the scale of the song) is the most
important variable, followed by root mean square energy, two coeddicients and 
rolloff.

Model is run again after scaling all variables
'''

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
## X_test is not included in the fitting for avoiding data leakage
x_test = scaler.transform(x_test)

forest = RandomForestClassifier(n_estimators = 1000)
forest.fit(x_train,y_train)

preds2 = forest.predict(x_test)

print(classification_report(y_test, preds2))
print(confusion_matrix(y_test, preds2))

plt.figure(figsize = (16,9))
sns.heatmap(confusion_matrix(y_test, preds2), xticklabels = labels,
            yticklabels = labels, annot = True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest on Scaled Data')

'''
Scaled values slightly did not improve the Random Forest Classifier's
performance


Tensorflow approach:
'''


'''
Modelcheckpoint was used to save highest val accuracy as code shows below
'''


mc = ModelCheckpoint('weights-{epoch:02d}-{val_loss:.2f}.hdf5',
                     monitor='val_accuracy',
                     save_best_only = True,)
    

losses = []
#for  lr in lrlist:
def createModel():
    ann = Sequential()
    ann.add(Dense(128, activation = 'relu'))
    ann.add(Dropout(0.25))
    ann.add(Dense(784, activation = 'relu'))
    ann.add(Dropout(0.25))
    ann.add(Dense(784, activation = 'relu'))
    ann.add(Dropout(0.25))
    ann.add(Dense(10, activation = 'softmax'))

    return ann


ann = createModel()
ann.compile(optimizer = 'adam',
                loss = ['sparse_categorical_crossentropy'],
                metrics = ['accuracy'])
ann.fit(x_train, y_train, epochs = 112,
        validation_data = (x_test,y_test),    
        batch_size = 10,
        #callbacks = [mc] #Uncomment for saving the model after improvement
        )

annf = tf.keras.models.load_model('saved-hdf5\\adam\weights-78-1.53.hdf5')
tfpreds1 = annf.predict_classes(x_test)

print(classification_report(y_test, tfpreds1))
print(confusion_matrix(y_test, tfpreds1))

plt.figure(figsize = (16,9))
sns.heatmap(confusion_matrix(y_test, tfpreds), xticklabels = labels,
            yticklabels = labels, annot = True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Tensorflow: Adam with default lr')
'''
Testing Adagrad optimizer
'''
ann = createModel()
ann.compile(optimizer = Adagrad(learning_rate = 0.02),
                loss = ['sparse_categorical_crossentropy'],
                metrics = ['accuracy'])
ann.fit(x_train, y_train, epochs = 112,
        validation_data = (x_test,y_test),    
        batch_size = 10,
        #callbacks = [mc] #Uncomment for saving the model after improvement
        )

annf = tf.keras.models.load_model('saved-hdf5\\adagrad\weights-62-1.24.hdf5')
tfpreds2 = annf.predict_classes(x_test)

print(classification_report(y_test, tfpreds2))
print(confusion_matrix(y_test, tfpreds2))

plt.figure(figsize = (16,9))
sns.heatmap(confusion_matrix(y_test, tfpreds), xticklabels = labels,
            yticklabels = labels, annot = True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Tensorflow: Adagrad with lr = 0.02')
