# coding: utf-8
import sys
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
imp = Imputer(strategy='mean', axis=0)
from sklearn.linear_model import Ridge, RidgeCV, BayesianRidge
from sklearn.metrics import mean_squared_error
from numpy import concatenate
from utils import *
TrainFiles = {'T':"./data/Temperature_Train_Feature.tsv", 'S':"./data/SunDuration_Train_Feature.tsv", 'P':"./data/Precipitation_Train_Feature.tsv"}
TestFiles = {'T':"./data/Temperature_Test_Feature.tsv", 'S':"./data/SunDuration_Test_Feature.tsv", 'P':"./data/Precipitation_Test_Feature.tsv"}
TrainTarget = "./data/Temperature_Train_Target.dat.tsv"
Location = "./data/Location.tsv"

def getFeature(nPrev, nAfter, aux_temp, aux_sun, aux_prec, files):
    temp_feature = pd.read_csv(files['T'], sep='\t')
    sun_feature = pd.read_csv(files['S'], sep='\t')
    prec_feature = pd.read_csv(files['P'], sep='\t')
    location = pd.read_csv(Location, sep='\t', header=None)

    X_t = temp_feature.loc[:, ['place%d' % i for i in range(11)]].values
    X_s = sun_feature.loc[:, ['place%d' % i for i in range(11)]].values
    X_p = prec_feature.loc[:, ['place%d' % i for i in range(11)]].values
    X_time = temp_feature.loc[:, ['hour', 'day']].values
    X_target = get_targetX(temp_feature.loc[:, ['targetplaceid']].values, 0)
    X_which = concatenate((auxiliary(X_t), auxiliary(X_s), auxiliary(X_p)), axis=1)

    l = location.loc[1:, 1:].values
    l_1800 = np.tile(l, (164, 1))[:-4,:]
    X_basic = concatenate((X_t, X_target, X_time, X_which, l_1800),axis=1)

    #X_Ngram = n_gram(nPrev,nAfter, X_basic, crossYearPt=getCrossYear(temp_train_feature))
    X_Ngram = n_gram(nPrev, nAfter, X_basic)
    
    X_tempAux = concatenate(map(lambda i:auxiliary(X_t, i), aux_temp), axis=1)
    X_sunAux = concatenate(map(lambda i:auxiliary(X_s, i), aux_sun), axis=1)
    X_precAux = concatenate(map(lambda i:auxiliary(X_p, i), aux_prec), axis=1)

    #X_dev = concatenate((np.std(X_t, axis=1).reshape(1800, 1)), axis=1)
    X_Ave = concatenate((n_average(X_t, 2), n_average(X_t, 4)), axis=1)

    X_Final = concatenate((X_Ngram, X_tempAux, X_sunAux, X_precAux, X_Ave), axis=1)
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
    SUBMIT_PATH = 'submission/submission_0707_2.dat'
    #np.savetxt(SUBMIT_PATH, y_test_pred, fmt='%.10f')

if __name__ == "__main__":
    '''
    MIN = 1
    WHICH = None
    for i in range(0, 12):
        for j in range(0, 12):
            print "%s,%s-gram" % (i, j)
            try:
                mse = validate(i,j, [-2, -1, 1, 2], [-1], [-1, -2])
                if mse < MIN:
                    MIN = mse
                    WHICH = (i,j)
            except:
                pass
    sys.stderr.write("the best is %s (mse:%f)\n" % (WHICH, MIN))
    '''
    model5_3 = validate(5, 3, [-2, -1, 1, 2], [-1], [-1, -2], get_model=True)
    predict(5, 3, [-2, -1, 1, 2], [-1], [-1, -2], model5_3)
