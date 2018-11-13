#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:45:32 2018
Regression Analysis
@author: xave
"""

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,BatchNormalization
from keras import regularizers
from keras.optimizers import Adam,SGD,RMSprop
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn import svm 
from collections import defaultdict
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn import svm
from collections import defaultdict
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.externals import joblib
from scipy.sparse import csr_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ridge_regression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score

DN=pd.read_csv("/Users/xave/Downloads/CPI-1022/balanced_cleaned.csv",sep="\t")
DN=DN.drop(["Unnamed: 0","Resolution","Unnamed: 0.1"],axis=1)
# data for regression
DR=DN[DN["Target"]==1]
####regression
DR.columns
DRX=DR.iloc[:,5:]
DRY=DR.iloc[:,0]
DRX=DRX.replace([np.inf, -np.inf], np.nan)
l22=DRX.mean(axis=0)
DRX=DRX.fillna(l22)


scaler=StandardScaler()
scaler.fit(DRX)
DRX=scaler.transform(DRX)
print("sample size:",DRX.shape[0])

X_train, X_test, y_train, y_test = train_test_split(DRX, DRY,test_size=0.2, random_state=2)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

###keras with batch_normalization


def dnn_reg(X_train,y_train,X_test,y_test):  
    model=Sequential()
    model.add(Dense(1000,input_dim=1583,kernel_initializer="glorot_normal",activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(1000,kernel_initializer="glorot_normal",activation="relu"))
    model.add(BatchNormalization())
# =============================================================================
#     model.add(Dropout(0.6))
# =============================================================================
    model.add(Dense(1,kernel_initializer="normal"))
    adam=Adam(lr=0.002)   # model.add(Dense(units=1,activation="relu",kernel_initializer="glorot_uniform"))
    sgd=SGD(lr=0.01, momentum=0.01, decay=0.0, nesterov=True)
    rms=RMSprop(lr=0.005)
    model.compile(optimizer=adam,loss="mean_squared_error")
    callbacks=[EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True)]
    print(model.summary())
    model.fit(X_train,y_train,batch_size=4,epochs=50,verbose=1,callbacks=callbacks)
    return (model)


def reg_eval(X_test,y_test,model):
    y_pred=model.predict(X_test)
    y_p=[]
    for i in y_pred:
        for j in i:
            y_p.append(j)
    print("evaluation the results for model:",model)
    print("MSE:",mean_squared_error(y_test,y_p))
    print("R2:",r2_score(y_test,y_p))
    print("EVS:",explained_variance_score(y_test,y_p))

md=dnn_reg(X_train,y_train,X_test,y_test)
reg_eval(X_test,y_test,md)

###Lasso CV regression

def reg_eval2(y_test,model):
    y_pred=model.predict(X_test)
    print("evaluation the results for model:",model)
    print("MSE:",mean_squared_error(y_test,y_pred))
    print("R2:",r2_score(y_test,y_pred))
    print("EVS:",explained_variance_score(y_test,y_pred))

lasso = LassoCV(cv=5, random_state=0,max_iter=10000)
lasso.fit(X_train,y_train)
reg_eval2(y_test,lasso)

#ElasticNet Regressionb
ela = ElasticNetCV(l1_ratio=0.8,normalize=True,max_iter=5000,random_state=77)
ela.fit(X_train,y_train)
print("R square:",ela.score(X_test,y_test))
reg_eval2(y_test,ela)


#SVR Regression
from sklearn.svm import LinearSVR
LSVR=LinearSVR(epsilon=0.1,random_state=0, tol=1e-5,max_iter=10000)
# scaler=RobustScaler()
# pipe=Pipeline(steps=[("scaling",scaler),("rg",LSVR)])
LSVR.fit(X_train,y_train)
reg_eval2(y_test,LSVR))
