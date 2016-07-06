# coding: utf-8
import pandas as pd
import numpy as np
from numpy import concatenate
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean', axis=0)

def getCrossYear(Temp_file):
    Year = Temp_file.loc[:, ['year']].values
    NowYear = -1
    crossYearPt = []
    for i in range(Year.shape[0]):
        if NowYear != Year[i]:
            NowYear = Year[i, 0]
            crossYearPt.append(i)
    return crossYearPt[1:]

def n_gram(n, m, M, crossYearPt=[], only=False):
    # get dummy.
    imp.fit(M)
    M = imp.transform(M)
    dummy = np.mean(M, axis=0)
    # get dummy.
    M_prev = M 
    M_ngram = M
    for i in range(n-1):
        M_prev = np.vstack([dummy, M_prev])[:-1,:] 
        for y in crossYearPt:
            M_prev[y] = dummy
        M_ngram = concatenate((M_ngram, M_prev), axis=1)
    if only and n != 0:
        return M_prev
    
    M_post = M
    for i in range(m-1):
        M_post = np.vstack([M_post, dummy])[1:,:]
        for y in crossYearPt:
            M_post[y] = dummy
        M_ngram = concatenate((M_ngram, M_post), axis=1)
    if only and m != 0:
        return M_post
    return M_ngram

def n_average(n, M):
    M_prev = M
    M_post = M
    M_window = M
    for i in range(n-1):
        M_prev = np.vstack([np.zeros(M.shape[1]), M_prev])[:-1,:] 
        M_post = np.vstack([M_post, np.zeros(M.shape[1])])[1:,:]
        M_window = concatenate((M_prev, M_window, M_post), axis=1)
    windowMean = np.mean(M_window, axis=1)
    return windowMean.reshape((1800, 1))

def auxiliary(M, which=0):
    # get target matrix.
    if which > 0:
        targetM = n_gram(0, which, M, only=True)
    elif which < 0:
        targetM = n_gram(which*(-1), 0,M, only=True)
    else:
        targetM = M
    temp = []
    for i in range(1800):
        which = i % 11         
        nowRow = targetM[i,:]
        nowElement = nowRow[which]
        temp.append(nowElement)
    temp = np.asarray(temp)
    return temp.reshape((1800, 1))
