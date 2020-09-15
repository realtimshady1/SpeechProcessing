# Importing required libraries
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob 
import os
import pickle
import IPython.display as ipd  # To play sound in the notebook

# Define datasets of interest
data_folder = ['OAF_happy/', 'YAF_happy/', 'OAF_Sad/', 'YAF_sad/' ]
data_group = ['happy', 'happy', 'sad', 'sad']
data_path = '../input/toronto-emotional-speech-set-tess/TESS Toronto emotional speech set data/'

# Take example and play it
filename = 'YAF_back_happy.wav'
sample_rate, samples = wavfile.read(data_path + data_folder[1] + filename)

# Play Audio
ipd.Audio(samples, rate=sample_rate)

# Convert to MFCC co-efficient
dataset = pd.DataFrame(columns=['feature'])
labels = []

# Cycle through each datafile
i = 0
j = 0
for group in data_folder:
    for filename in os.listdir(data_path + group):
        samples, sample_rate = librosa.load(data_path + group + filename)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=13), axis=0)

        labels.append(data_group[j])
        dataset.loc[i] = [mfccs]
        i = i + 1
    j = j + 1

# Convert dataset to array and fill blanks
dataset = pd.DataFrame(dataset['feature'].values.tolist())
dataset = dataset.fillna(0)

# Create training sets and test sets
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.3, shuffle=True)

# Preprocess data by Gaussian normalization
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean)/std
mean = np.mean(x_test, axis=0)
std = np.std(x_test, axis=0)
x_test = (x_test - mean)/std

# Convert to array
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Convert to numeric labels
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

# Pickel the lb object for future use 
filename = 'labels'
outfile = open(filename,'wb')
pickle.dump(lb,outfile)
outfile.close()

# Create additional axis for convolution
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

# New model
model = Sequential()
model.add(Conv1D(128, 8, padding='same', activation='relu', input_shape=(x_train.shape[1],1)))
model.add(Conv1D(128, 8, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Conv1D(64, 8, padding='same', activation='relu'))
model.add(Conv1D(64, 8, padding='same', activation='relu'))
model.add(Conv1D(64, 8, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# Train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_history=model.fit(x_train, y_train, batch_size=16, epochs=25, validation_data=(x_test, y_test))

# Plot the training results
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Evaluate the model accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))