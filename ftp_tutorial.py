import sys
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
TEMPERATURE_TRAIN_FEATURE_PATH = 'Temperature_Train_Feature.tsv'
TEMPERATURE_TRAIN_TARGET_PATH = 'Temperature_Train_Target.dat.tsv'
TEMPERATURE_TEST_FEATURE_PATH = 'Temperature_Test_Feature.tsv'
LOCATION = 'Location.tsv'
location = pd.read_csv(LOCATION, sep='\t')
height = location.loc[:, 'height'].values 

imp = Imputer(strategy='mean', axis=0)

def train_model(xTrain, yTrain):
    # process data
    goal = xTrain.shape
    h_feat = np.asarray(map(lambda x:height[x] * 0.01, xTrain[:,-1]))
    xTrain = np.append(xTrain[:,:-1], h_feat)
    xTrain = xTrain.reshape(goal)
    sys.exit()
    reg = Ridge(alpha=0.1)
    reg.fit(xTrain, yTrain)
    return reg

### test by splitted data. ###
def model_test():
    # read training data.
    data_train_feature = pd.read_csv(TEMPERATURE_TRAIN_FEATURE_PATH, sep='\t')
    which = ['place%d' % i for i in range(11)] + ['targetplaceid'] 
    X = data_train_feature.loc[:, which].values
    y = np.loadtxt(TEMPERATURE_TRAIN_TARGET_PATH)
    # split.
    TEST_SIZE = 0.2
    RANDOM_STATE = 0
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # fix nan.
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    imp.fit(X_val)
    X_val = imp.transform(X_val)
    # train_model
    MD = train_model(X_train, y_train)
    # predict.
    y_val_pred = MD.predict(X_val)
    MSE = mean_squared_error(y_val, y_val_pred)
    print MSE

def print_submit():
    imp.fit(X)
    X = imp.transform(X)
    MD = train_model(X, y)

    data_test_feature = pd.read_csv(TEMPERATURE_TEST_FEATURE_PATH, sep='\t')
    X_test = data_test_feature.loc[:, ['place%d' % i for i in range(11)]].values
    X_test = imp.transform(X_test)
    y_test_pred = MD.predict(X_test)
    # print to file.
    SUBMIT_PATH = 'submission.dat'
    np.savetxt(SUBMIT_PATH, y_test_pred, fmt='%.10f')

if __name__ == "__main__":
    model_test()
