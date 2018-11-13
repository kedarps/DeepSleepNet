# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:47:17 2017

Module for importing Training and Testing Data from .mat files in Matlab. 
Ensure to save .mat files using the '-v7.3' flag since this usesh5py package

@author: ksp6
"""
import h5py
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

def getTrainingData(file_name, one_hot_encode=True, min_max_scale=False):
    print('Reading mat file...')
    # read -v7.3 .mat files using h5py, make sure to save them using this flag in matlab
    # this will give array of arrays
    with h5py.File(file_name, 'r') as f:
        X_train = np.asarray([ f[element][:].transpose() for element in f['data']['X_train'][()][0] ])
        Y_train = np.asarray([ f[element][:].transpose().ravel() for element in f['data']['Y_train'][()][0] ])
    
    # concatenate into large arrays
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)
    
    print('Consolidating classes N3 and N4...')
    # get rid of class 3, merge it with class 4
    Y_train[Y_train == 3] = 4
    
    if min_max_scale:
        print('Standardizing dataset...')
        X_train = scale_data(X_train)
    
    if one_hot_encode:
        print('Running one-hot-encoding...')
        # one hot encoding for output labels
        Y_train = one_hot_encode_data(Y_train.reshape(-1, 1))
            
    return (X_train, Y_train)

def appendZeros(inData, num_zeros=10):
    outData = np.append(inData, np.zeros((inData.shape[0], num_zeros)), axis=1)
    
    return outData
    

def getTestingData(file_name, one_hot_encode=True, min_max_scale=False):
    print('Reading mat file...')
    # read -v7.3 .mat files using h5py, make sure to save them using this flag in matlab
    with h5py.File(file_name, 'r') as f:
        X_test = f['data']['X_test'][()].transpose()
        Y_test = f['data']['Y_test'][()].transpose().ravel()
        
        # testing index, it is corrected for python's zero-indexing
        testIdx = f['data']['TestSubIdx'][()] - 1
        
    print('Consolidating classes N3 and N4...')
    Y_test[Y_test == 3] = 4
    
    if min_max_scale:
        print('Standardizing dataset...')
        X_test = scale_data(X_test)
    
    if one_hot_encode:
        print('Running one-hot-encoding...')
        # one hot encoding for output labels
        Y_test = one_hot_encode_data(Y_test.reshape(-1, 1))
        
    return (X_test, Y_test), testIdx
    
def scale_data(inData):
    scaler = MinMaxScaler()
    outData = scaler.fit_transform(inData)
    
    return outData

def one_hot_encode_data(inData):
    ohe = OneHotEncoder()
    outData = ohe.fit_transform(inData).toarray()
    
    return outData

def oversample_minority_class(X, Y):
    print('Oversampling minority classes based on number of instances in Class 5...')
    
    # this does one hot decoding, but since we removed class 3, we only have 5 classes
    # and now after argmax, 3 corresponds to class 5
    Y = np.argmax(Y, axis=1)
    
    # we will oversample number of samples in class 5, which corresponds to 3 in 
    # OHD Y
    majClass = 3
    nObsPerClass = sum( Y == majClass )
    nFeats = X.shape[1]
    
    # initialize empty arrays with (num_5 * 6) observations
    X_out = np.empty( (nObsPerClass*6, nFeats) )
    Y_out = np.empty( (nObsPerClass*6,) )
    
    ptr = 0
    for l in np.unique(Y):
        X_here = X [ Y == l,: ]
        
        # keep in mind, now 3 corresponds to class 5
        if l == majClass:
            X_here_ovs = X_here
        else:
            X_here_ovs = resample(X_here, n_samples=nObsPerClass)
        
        X_out[ ptr:ptr+nObsPerClass,: ] = X_here_ovs
        Y_out[ ptr:ptr+nObsPerClass ] = l * np.ones((nObsPerClass,))
        
        ptr += nObsPerClass
    
    Y_out = one_hot_encode_data(Y_out.reshape(-1, 1))
    return (X_out, Y_out)
    