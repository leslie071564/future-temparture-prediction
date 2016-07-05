# coding: utf-8
import sys
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean', axis=0)
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from numpy import concatenate
TrainFiles = {'T':"./data/Temperature_Train_Feature.tsv", 'S':"./data/SunDuration_Train_Feature.tsv", 'P':"./data/Precipitation_Train_Feature.tsv"}
TestFiles = {'T':"./data/Temperature_Test_Feature.tsv", 'S':"./data/SunDuration_Test_Feature.tsv", 'P':"./data/Precipitation_Test_Feature.tsv"}
TrainTarget = "./data/Temperature_Train_Target.dat.tsv"
Location = "./data/Location.tsv"

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

def getFeature(nPrev, nAfter, aux_temp, aux_sun, aux_prec, files):
    temp_feature = pd.read_csv(files['T'], sep='\t')
    sun_feature = pd.read_csv(files['S'], sep='\t')
    prec_feature = pd.read_csv(files['P'], sep='\t')
    location = pd.read_csv(Location, sep='\t', header=None)

    X_t = temp_feature.loc[:, ['place%d' % i for i in range(11)]].values
    X_s = sun_feature.loc[:, ['place%d' % i for i in range(11)]].values
    X_p = prec_feature.loc[:, ['place%d' % i for i in range(11)]].values
    X_aux = temp_feature.loc[:, ['targetplaceid', 'hour', 'day']].values
    X_which = concatenate((auxiliary(X_t), auxiliary(X_s), auxiliary(X_p)), axis=1)

    l = location.loc[1:, 1:].values
    l_1800 = np.tile(l, (164, 1))[:-4,:]
    X_basic = concatenate((X_t, X_aux, X_which, l_1800),axis=1)

    #X_Ngram = n_gram(nPrev,nAfter, X_basic, crossYearPt=getCrossYear(temp_train_feature))
    X_Ngram = n_gram(nPrev, nAfter, X_basic)
    
    X_tempAux = concatenate(map(lambda i:auxiliary(X_t, i), aux_temp), axis=1)
    X_sunAux = concatenate(map(lambda i:auxiliary(X_s, i), aux_sun), axis=1)
    X_precAux = concatenate(map(lambda i:auxiliary(X_p, i), aux_prec), axis=1)
    X_Final = concatenate((X_Ngram, X_tempAux, X_sunAux, X_precAux), axis=1)
    return X_Final

def validate(nPrev, nAfter, aux_temp, aux_sun, aux_prec, get_model=False):
    X_Final = getFeature(nPrev, nAfter, aux_temp, aux_sun, aux_prec, TrainFiles)
    data_train_target = pd.read_csv(TrainTarget, sep='\t', header=None)
    y = data_train_target.loc[:,0].values

    TEST_SIZE = 0.2
    RANDOM_STATE = 0
    X_train, X_val, y_train, y_val = train_test_split(X_Final, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    imp.fit(X_train)
    X_train = imp.transform(X_train)
    imp.fit(X_val)
    X_val = imp.transform(X_val)

    reg = RidgeCV()
    reg.fit(X_train, y_train)
    y_val_pred = reg.predict(X_val)
    print mean_squared_error(y_val, y_val_pred)
    
    if get_model:
        imp.fit(X_Final)
        X_Final = imp.transform(X_Final)
        reg_submit = RidgeCV()
        reg_submit.fit(X_Final, y)
        return reg_submit
    return mean_squared_error(y_val, y_val_pred)

def predict(nPrev, nAfter, aux_temp, aux_sun, aux_prec, model):
    X_Final = getFeature(nPrev, nAfter, aux_temp, aux_sun, aux_prec, TestFiles)
    imp.fit(X_Final)
    X_Final = imp.transform(X_Final)

    y_test_pred = model.predict(X_Final)
    SUBMIT_PATH = 'submission/submission_0705_3.dat'
    np.savetxt(SUBMIT_PATH, y_test_pred, fmt='%.10f')

if __name__ == "__main__":
    '''
    MIN = 1
    WHICH = None
    for i in range(2, 12):
        for j in range(2, 12):
            print "%s,%s-gram" % (i, j)
            mse = validate(i,j)
            if mse < MIN:
                MIN = mse
                WHICH = (i,j)
    sys.stderr.write("the best is %s (mse:%f)\n" % (WHICH, MIN))
    '''
    model10_3 = validate(10, 3, [-2, -1, 1, 2], [-1], [-1, -2], get_model=True)
    predict(10, 3, [-2, -1, 1, 2], [-1], [-1, -2], model10_3)
