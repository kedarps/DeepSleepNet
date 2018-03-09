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
import scipy.io as io

# =============================================================================
# Script
# =============================================================================
# undersampling factor, usr=1 corresponds to no undersampling
usr = '1'
ch = '1'

# path where files live
dataDir = os.path.join(os.getcwd(), 'data', 'time_domain', 'USR_'+usr, 'CH_'+ch)
seqTrainDir = os.path.join(os.getcwd(), 'seqTrainResults')
files = os.listdir(dataDir)

for f in files[:1]:
    # data file
    datFile = os.path.join(dataDir, f)
    
    (X_test, Y_test), _ = mat.getTestingData(datFile)
    testIdx = (f.split('.')[0]).split('_')[1]
    print('Testing on sub {}...\n'.format(testIdx))
    
    # reshape to keep input to NN consistent
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    (_, n_feats, _) = X_test.shape
    (_, n_classes) = Y_test.shape
    
    # first build pre-train net
    preTrainNN = dsn.preTrainingNet(n_feats, n_classes)
    seqTrainNN = dsn.fineTuningNet(n_feats, n_classes, preTrainNN)
    
    # load network weights from seq training results
    seqFile = os.path.join(seqTrainDir, 'superviseSeqTrainNet_TestSub'+testIdx+'.h5')
    seqTrainNN.load_weights(seqFile, by_name=True)
    
    Y_pred = seqTrainNN.predict(X_test, verbose=1)
    io.savemat('predSeqTestSub'+testIdx+'.mat', {'Y_pred' : Y_pred, 'Y_true' : Y_test})
    
    