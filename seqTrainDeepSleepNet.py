# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 23:30:24 2017
takes supervised pre-trained network and appends the sequential training network to train on sequential data

@author: ksp6
"""

import os
import numpy as np
import readMat as mat
import deepSleepNet as dsn

# =============================================================================
# Script
# =============================================================================
# undersampling factor, usr=1 corresponds to no undersampling
usr = '1'
ch = '1'

# path where files live
dataDir = os.path.join(os.getcwd(), 'data', 'time_domain', 'USR_'+usr, 'CH_'+ch)
preTrainDir = os.path.join(os.getcwd(), 'preTrainResults')
files = os.listdir(dataDir)

for f in files:
    # data file
    datFile = os.path.join(dataDir, f)
    
    (X_train_seq, Y_train_seq) = mat.getTrainingData(datFile)
    testIdx = (f.split('.')[0]).split('_')[1]
    
    # reshape to keep input to NN consistent
    X_train_seq = np.reshape(X_train_seq, (X_train_seq.shape[0], X_train_seq.shape[1], 1))
    
    (n_samples, n_feats, _) = X_train_seq.shape
    (_, n_classes) = Y_train_seq.shape
    
    # first build pre-train net
    preTrainNN = dsn.preTrainingNet(n_feats, n_classes)
    # load weights from saved network
    psrFile = os.path.join(preTrainDir, 'supervisePreTrainNet_TestSub'+testIdx+'.h5')
    preTrainNN.load_weights(psrFile, by_name=True)
    
    seqTrainNN = dsn.fineTuningNet(n_feats, n_classes, preTrainNN)
    
    
    