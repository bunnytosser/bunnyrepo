#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:02:29 2018

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


from sklearn import svm 
from collections import defaultdict
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn import svm
from collections import defaultdict
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.externals import joblib
from scipy.sparse import csr_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


precision_scorer=metrics.make_scorer(metrics.precision_score)
auc_scorer=metrics.make_scorer(metrics.roc_auc_score)
def evaluation(y_test,y_pred,class1,X_test,X_train):
    confusion_mat = confusion_matrix(y_test.tolist(), y_pred.tolist())
    print(confusion_mat)
    print('Accuracy of classifier on test set: {:.2f}'.format(class1.score(X_test, y_test)))
    print("classification report:",classification_report(y_test, y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    print ('KS:',max(tpr-fpr))
    print("test set positive size:")
    print(y_pred.sum())
    print("precision score is:",precision_score(y_test,y_pred))
    print("AUC is:",metrics.auc(fpr, tpr))
    y_trainpred=class1.predict(X_train)
    print("training set precision score:",precision_score(y_train,y_trainpred))
    print("training set recall score:",recall_score(y_train,y_trainpred))
    print("training set AUC score:",metrics.roc_auc_score(y_train,y_trainpred))
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_trainpred, pos_label=1)
    print('Accuracy of the model on training:',metrics.accuracy_score(y_train,y_trainpred))
    print ('train set KS:',max(tpr-fpr))
    print("classification report:",classification_report(y_train, y_trainpred))
    
    
DN=pd.read_csv("balanced_cleaned.csv",sep="\t")
DN=DN.drop(["Unnamed: 0","Resolution","Unnamed: 0.1"],axis=1)
X=DN.iloc[:,5:]
y=DN.iloc[:,4]
print(np.any(np.isnan(X)))
print(np.any(np.isfinite(X)))


X=X.replace([np.inf, -np.inf], np.nan)
l22=X.mean(axis=0)
X=X.fillna(l22)

scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
print("sample size:",X.shape[0])
print("1:",sum(y[y==1]),"0:",X.shape[0]-sum(y[y==1]))
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=2)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# =============================================================================
# #a rf classifier
# clf = RandomForestClassifier(max_depth=4,max_features="log2",class_weight="balanced",bootstrap=True,random_state=99,n_estimators=1000,min_samples_split=2)
# clf.fit(X_train, y_train)
# y_pred=clf.predict(X_test)
# evaluation(y_test,y_pred,clf,X_test,X_train)
# 
# 
# auc_scorer=metrics.make_scorer(metrics.roc_auc_score)
# 
# #ind_params = max_depth=5,'learning_rate': 0.1, 'n_estimators': 120, 'random_state':0, 'subsample': 0.8,"max_features":"log2"}
# optimized_GBM=GradientBoostingClassifier(max_depth=3,learning_rate=0.08, n_estimators=1000, random_state=99, subsample=0.8,verbose=1,max_features="log2") 
#                             
# 
# optimized_GBM.fit(X_train, y_train)
# =============================================================================


def evaluation2(y_test,y_pred,class1,X_test,X_train,y_train):
# =============================================================================
#     y_train=pd.to_numeric(y_train)
#     y_test=pd.to_numeric(y_test)
# =============================================================================
  #  y_pred=pd.to_numeric(y_pred)

    confusion_mat = confusion_matrix(y_test, y_pred)
    print(confusion_mat)
    print("classification report:",classification_report(y_test, y_pred))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    print ('ACC:',accuracy_score(y_test,y_pred))
    print("test set positive size:")
    print(y_pred.sum())
    print("precision score is:",precision_score(y_test,y_pred))
    print("AUC is:",metrics.auc(fpr, tpr))
    y_trainpred=class1.predict_classes(X_train)
    print("training set precision score:",precision_score(y_train,y_trainpred))
    print("training set recall score:",recall_score(y_train,y_trainpred))
    print("training set AUC score:",metrics.roc_auc_score(y_train,y_trainpred))
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_trainpred, pos_label=1)
    print ('train set KS:',max(tpr-fpr))
    print("classification report:",classification_report(y_train, y_trainpred))
def dnn_class(X_train,y_train,X_test,y_test):
    
    model=Sequential()
    model.add(Dense(1000,input_dim=1583,kernel_initializer="glorot_uniform",activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(1000,kernel_initializer="glorot_uniform",activation="relu"))
    model.add(BatchNormalization())
# =============================================================================
#     model.add(Dropout(0.6))
# =============================================================================
    model.add(Dense(1,activation="sigmoid",kernel_initializer="glorot_uniform"))
    adam=Adam(lr=0.01)   # model.add(Dense(units=1,activation="relu",kernel_initializer="glorot_uniform"))
    sgd=SGD(lr=0.01, momentum=0.01, decay=0.0, nesterov=True)
    rms=RMSprop(lr=0.005)
    model.compile(optimizer=adam,loss="binary_crossentropy",metrics=["accuracy"])
    callbacks=[EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
    print(model.summary())
    model.fit(X_train,y_train,batch_size=4,epochs=50,verbose=1,callbacks=callbacks,validation_data=(X_test,y_test))
    pkl_filename = "keras_model.joblib"
    with open(pkl_filename, 'wb') as file:
        joblib.dump(model, file)
    y_pred=model.predict_classes(X_test)
    print(y_pred)
    evaluation2(y_test,y_pred,model,X_test,X_train,y_train)

dnn_class(X_train,y_train,X_test,y_test)
