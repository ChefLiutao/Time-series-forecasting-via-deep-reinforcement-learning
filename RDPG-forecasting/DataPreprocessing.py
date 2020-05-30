# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:00:32 2020

@author: ChefLiutao

This part of code is to load and preprocess time series data.
"""

import numpy as np


def build_s_a(sequence,n,m):
    '''
    Args:
        sequence: Time series data
        n: The number of historical data denoting the current state
        m: The number of prediction steps in advance
    Return:
        state_mat: A matrix contains all states at each time step
        best_action: The optimal action based on each state
    '''
    n_rows = len(sequence)-n-m+1
    state_mat = np.zeros((n_rows,n))
    best_action = np.zeros(n_rows)
    for i in range(n_rows):
        state_mat[i] = sequence[i:(i+n)]
        best_action[i] = sequence[i+n+m-1]
    return state_mat,best_action



def normalization(traindata,testdata):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(traindata)
    traindata_scaled = scaler.transform(traindata)
    testdata_scaled = scaler.transform(testdata)
    
    return traindata_scaled,testdata_scaled
