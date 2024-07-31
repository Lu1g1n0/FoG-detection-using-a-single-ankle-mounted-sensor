import csv
import numpy as np
import pandas as pd
from keras.metrics import AUC
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

'''
This script trains and test a one-dimensional convolutional neural network developed for freezing of gait detection from inertial data.

INPUT
The script takes in input two CSV files, containing train ('train_data.csv') and test ('test_data.csv') data.
These files should contain a table with N samples and 4 columns.
N samples dependent on the amount of data. Data should be sampled at 60 Hz
The 4 columns contains angular velocity data measured in degree per second (gyrV,gyrAP,gyrML) and the fog label (fogLabel)

OUTPUT
The script outputs the test label and prediction, writing in two CSV files ('test_label.csv' and 'test_prediction.csv')
The test label is extracted from the 'test_data.csv' file.
The prediction is obtained from the trained model, and is in form of probability (from 0 to 1).
Test label and prediction can then be used for computing classification metrics.
When evaluating test performance, remember that test data and label are here segmented with 75% overlap.

IMPORTANT INFORMATION
(1) Make sure your data is sampled or resampled to 60 Hz
(2) Adjust the learning rate, number of epochs, and batch size based on your dataset size. 
    Specifically, as your dataset size increases, reduce learning rate, and increase the number of epochs and batch size.
    This configuration should work well with 30 minutes to 2 hours of data
(3) While the window size should be fixed at 2 seconds, you can adjust the overlap as you prefer. 
    As the overlap increases, more windows are generated. This produces more training data, which is beneficial.
    However, too large overlap may lead to over-fitting. Thus, find the best compromise.
(4) This model has shown to provide good performance on acceleration data recorded from the waist/lower back.
    Evaluations on data recorded from other body locations have not been performed yet.

'''

def load_data(fileName):
    data = pd.read_csv(fileName)
    return data

def segmentData(data, w, o, Fs):
    windowSize, overlap = int(w * Fs), int(o * w * Fs)
    slide = windowSize-overlap
    total_windows = int(np.ceil((len(data) - windowSize) / slide))
    segmentedData = np.zeros((total_windows, windowSize, 3))
    segmentedLabels = np.zeros((total_windows,), dtype=int)
    for i in range(total_windows):
        start_idx = i * slide
        end_idx = start_idx + windowSize
        window_data = data.iloc[start_idx:end_idx, :3]
        window_data -= np.mean(window_data, axis=0)
        segmentedData[i] = window_data
        most_common_label = data.iloc[start_idx:end_idx, -1].mode().values[0]
        segmentedLabels[i] = most_common_label
    return segmentedData, segmentedLabels

def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    return loaded_model

def compile_model(model,lr,wd):
    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    pr_metric = AUC(curve='PR', num_thresholds=100)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[pr_metric])
    return model

def fit_model(model,windows,label):
    stop_early = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=0, mode='auto',restore_best_weights=True)
    X_train, X_val, y_train, y_val = train_test_split(windows, label, test_size=0.4, stratify=label)
    history = model.fit([X_train,X_train,X_train], y_train, epochs=120, batch_size=64, validation_data=([X_val,X_val,X_val], y_val))
    return model

if __name__ == '__main__':
    print('Loading training data')
    data = load_data('train_data.csv') # load data
    print('Segmenting data')
    windows,label = segmentData(data, w=2, o=0.75, Fs=60) # segment data: w = window size (in seconds, it should be 2 seconds), o = overlap (in percentage, 0.75 is 75%), Fs = sampling frequency (it should be 60 Hz)
    print('Loading model --> ', end='')
    model = load_model()  # load model
    print('compiling --> ', end='')
    model = compile_model(model,lr=0.008,wd=0.0004) # compile model: lr = learning rate, wd = weight decay
    print('fitting')
    model = fit_model(model,windows,label) # fit model
    print('Loading test data')
    testData = load_data('test_data.csv') # Load test data
    print('Segmenting test data')
    testData,testLabel = segmentData(testData, w=2, o=0.75, Fs=60) # segment test data
    print('Predicting...')
    prediction = model.predict([testData,testData,testData]) # predict test data
    print('Saving test label and prediction')
    prediction.tofile('test_prediction.csv', sep=',') # save prediction
    testLabel.tofile('test_label.csv',sep=',') # save label
    print('Done')