# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 06:43:58 2018
Test DeepSleepNet on testing data

@author: ksp6
"""

import os
import numpy as np
import deepSleepNet as dsn
import readMat as mat
import scipy.io as io

usr = '1'
ch = '1'

# path where files live
dataDir = os.path.join(os.getcwd(), 'data', 'time_domain', 'USR_'+usr, 'CH_'+ch)
nnDir = os.path.join(os.getcwd(), 'preTrainResults')
files = os.listdir(dataDir)

accuracies = []

for f in files[0:5]:
    file = os.path.join(dataDir, f)
    
    (X_test, Y_test), _ = mat.getTestingData(file)
    testIdx = (f.split('.')[0]).split('_')[1]
    print('Testing on sub {}...\n'.format(testIdx))
    
    # reshape to keep input to NN consistent
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    (_, n_feats, _) = X_test.shape
    (_, n_classes) = Y_test.shape
    
    # initialize deep sleep net
    preTrainNN = dsn.preTrainingNet(n_feats, n_classes)
    
    # set weights from pre-trained network
    psrName = os.path.join(nnDir, 'supervisePreTrainNet_TestSub'+testIdx+'.h5')
    preTrainNN.load_weights(psrName, by_name=True)
    
    Y_pred = preTrainNN.predict(X_test, verbose=1)
    io.savemat('predictionTestSub'+testIdx+'.mat', {'Y_pred' : Y_pred, 'Y_true' : Y_test})
    # now with saved network run testing data
#    scores = preTrainNN.evaluate(X_test, Y_test, verbose=1)
#    accuracies.append([int(testIdx), scores[1]*100])
#    print('Accuracy = {} %...\n'.format(scores[1]*100))
#    
#np.savetxt('preTrainTestingAccuracies.csv', np.array(accuracies), delimiter=';')
    