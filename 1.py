import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")


Xmat=scipy.io.loadmat('input.mat')
ymat=scipy.io.loadmat('output.mat')

X=pd.DataFrame(data=Xmat['input'].T)
y=pd.DataFrame(data=ymat['output'].T)

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=0)


def get_score(valid_data,pred_data):
    mse=mean_squared_error(valid_data, pred_data)
    r2=r2_score(valid_data, pred_data)
    mae=mean_absolute_error(valid_data, pred_data)
    mape=mean_absolute_percentage_error(valid_data,pred_data)
    return (mse,r2,mae,mape)


hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1','l2'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': -1,
    "max_depth": 7,
    "num_leaves": 128,
    "max_bin": 512,
    "num_iterations": 100000
}

callbacks = [lgb.early_stopping(10000)]

gbm = lgb.LGBMRegressor(**hyper_params)

light_pred=y_test.copy()
for i in range(y.shape[1]):
    gbm.fit(X_train, y_train[i],
            eval_set=[(X_test, y_test[i])],
            eval_metric='l1',
            callbacks=callbacks)

    light_pred[i]=gbm.predict(X_test)

light_pred=pd.DataFrame(data=light_pred)

print(get_score(y_test, light_pred))