# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 23:30:24 2017
Trains neural network on oversampled training data

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
files = os.listdir(dataDir)

for f in files:
    file = os.path.join(dataDir, f)
    
    (X_train, Y_train) = mat.getTrainingData(file)
    (X_train_ovs, Y_train_ovs) = mat.oversample_minority_class(X_train, Y_train)
    testIdx = (f.split('.')[0]).split('_')[1]
    
    # reshape to keep input to NN consistent
    X_train_ovs = np.reshape(X_train_ovs, (X_train_ovs.shape[0], X_train_ovs.shape[1], 1))
    
    (n_samples, n_feats, _) = X_train_ovs.shape
    (_, n_classes) = Y_train_ovs.shape
    
    # pre-training phase
    preTrain = dsn.preTrainingNet(n_feats, n_classes)
    # train this network on oversampled dataset
    preTrain.fit(X_train_ovs, Y_train_ovs, epochs=75, batch_size=100)
    
    # save neural network weights so that we can use them while testing    
    preTrain.save_weights('supervisePreTrainNet_TestSub'+ testIdx +'.h5')