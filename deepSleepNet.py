# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:37:39 2018

@author: ksp6
"""

from keras.models import Model
from keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten, LSTM, Input, concatenate, Reshape
from keras.layers.wrappers import Bidirectional

# sampling rate
Fs = 100

def makeConvLayers(inputLayer):
    # two conv-nets in parallel for feature learning, 
    # one with fine resolution another with coarse resolution    
    # network to learn fine features
    convFine = Conv1D(filters=64, kernel_size=int(Fs/2), strides=int(Fs/16), padding='same', activation='relu', name='fConv1')(inputLayer)
    convFine = MaxPool1D(pool_size=8, strides=8, name='fMaxP1')(convFine)
    convFine = Dropout(rate=0.5, name='fDrop1')(convFine)
    convFine = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', name='fConv2')(convFine)
    convFine = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', name='fConv3')(convFine)
    convFine = Conv1D(filters=128, kernel_size=8, padding='same', activation='relu', name='fConv4')(convFine)
    convFine = MaxPool1D(pool_size=4, strides=4, name='fMaxP2')(convFine)
    fineShape = convFine.get_shape()
    convFine = Flatten(name='fFlat1')(convFine)
    
    # network to learn coarse features
    convCoarse = Conv1D(filters=32, kernel_size=Fs*4, strides=int(Fs/2), padding='same', activation='relu', name='cConv1')(inputLayer)
    convCoarse = MaxPool1D(pool_size=4, strides=4, name='cMaxP1')(convCoarse)
    convCoarse = Dropout(rate=0.5, name='cDrop1')(convCoarse)
    convCoarse = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', name='cConv2')(convCoarse)
    convCoarse = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', name='cConv3')(convCoarse)
    convCoarse = Conv1D(filters=128, kernel_size=6, padding='same', activation='relu', name='cConv4')(convCoarse)
    convCoarse = MaxPool1D(pool_size=2, strides=2, name='cMaxP2')(convCoarse)
    coarseShape = convCoarse.get_shape()
    convCoarse = Flatten(name='cFlat1')(convCoarse)
    
    # concatenate coarse and fine cnns
    mergeLayer = concatenate([convFine, convCoarse], name='merge')
    
    return mergeLayer, (coarseShape, fineShape)

def preTrainingNet(n_feats, n_classes):
    inLayer = Input(shape=(n_feats, 1), name='inLayer')
    mLayer, (_, _) = makeConvLayers(inLayer)
    outLayer = Dense(n_classes, activation='softmax', name='outLayer')(mLayer)
    #outLayer = Dense(n_feats, activation='sigmoid', name='outLayer')(mLayer)
    
    network = Model(inLayer, outLayer)
    network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #network.compile(loss='mean_squared_error', optimizer='adadelta')
    
    return network
    
def fineTuningNet(n_feats, n_classes):
    inLayer = Input(shape=(n_feats, 1), name='inLayer')
    mLayer, (cShape, fShape) = makeConvLayers(inLayer)
    outLayer = Dropout(rate=0.5, name='mDrop1')(mLayer)
    
    # this is the network that learns temporal dependencies using LSTM
    # merge the outputs of last layers
    # reshape because LSTM layer needs 3 dims (None, 1, n_feats)
    outLayer = Reshape((1, int(fShape[1]*fShape[2] + cShape[1]*cShape[2])))(outLayer)
    outLayer = Bidirectional(LSTM(512, activation='relu', dropout=0.5, name='bLstm1'))(outLayer)
    outLayer = Reshape((1, int(outLayer.get_shape()[1])))(outLayer)
    outLayer = Bidirectional(LSTM(512, activation='relu', dropout=0.5, name='bLstm2'))(outLayer)
    outLayer = Dense(n_classes, activation='softmax', name='outLayer')(outLayer)
    
    network = Model(inLayer, outLayer)
    network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return network
