#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import pdb
import sys
import copy
import pandas as pd
from scipy import stats
from scipy.stats import norm
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from collections import defaultdict
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MaxAbsScaler
import pickle
from sklearn.externals import joblib
from scipy.sparse import csr_matrix
from sklearn.ensemble import GradientBoostingClassifier
from ast import literal_eval
import mysql.connector as sql
import warnings
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore', 'Solver terminated early.*') 
import logging
import logging.config

logger = logging.getLogger("LOG")

class train_model:
    def __init__(self):
        pass

    def evaluation2(self,y_test,y_pred,class1,X_test,X_train,y_train):
        confusion_mat = confusion_matrix(y_test, y_pred)
        logger.info(confusion_mat)
        logger.info("classification report:",classification_report(y_test, y_pred))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        logger.info('KS: %s'%max(tpr-fpr))
        logger.info("test set positive size:")
        logger.info(y_pred.sum())
        logger.info("precision score is:%s" %precision_score(y_test,y_pred))
        logger.info("AUC is: %s" %metrics.auc(fpr, tpr))
        y_trainpred=class1.predict(X_train)
        y_train=pd.to_numeric(y_train)
        y_trainpred=pd.to_numeric(y_trainpred)
        logger.info("training set precision score: %s"%precision_score(y_train,y_trainpred))
        logger.info("training set recall score: %s"%recall_score(y_train,y_trainpred))
        logger.info("training set AUC score %s"%metrics.roc_auc_score(y_train,y_trainpred))
        fpr, tpr, thresholds = metrics.roc_curve(y_train, y_trainpred, pos_label=1)
        logger.info ('train set KS: %s'%max(tpr-fpr))
        logger.info("classification report: %s"%classification_report(y_train, y_trainpred))

    def evaluation(self,target,y_test,y_pred,class1,X_test,X_train,y_train,models,y_p):
        results={}
        results["t_name"]=target
        results["t_model"]=models
        results["ACC"]=metrics.accuracy_score(y_test, y_pred)
        results["recall"]=metrics.recall_score(y_test, y_pred)
        results["F1"]=metrics.f1_score(y_test, y_pred)
        c_k=["tn", "fp", "fn", "tp"]
        c_v=confusion_matrix(y_test, y_pred).ravel().tolist()
        results["confusion_matrix"]=dict(zip(c_k,c_v))
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
        results["AUC"]=metrics.auc(fpr, tpr)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_p, pos_label=1)
        pairs=[]
        for i in range(1,min(len(fpr),14)):
            indexer=round(i*len(fpr)/15)
            pairs.append([fpr[indexer],tpr[indexer]])
        results["ROC"]=pairs
        return(results)
        
    def randomforest_class(self,target,X_train,y_train,X_test,y_test,n_estimators=150,min_samples_split=5,max_depth=5):
        models="rf"
        scaler=MaxAbsScaler()
        clf = RandomForestClassifier(max_features="log2",class_weight="balanced",verbose=1,bootstrap=True,random_state=54,n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split)
        grid=Pipeline(steps=[("scaling",scaler),("clf",clf)])
        grid.fit(X_train, y_train)
        pkl_filename = "workspace/test_user/qsar_models/"+target+"_rf_model.joblib"
        with open(pkl_filename, 'wb') as file:
            joblib.dump(grid, file)
        y_pred = grid.predict(X_test)
        y_p = grid.predict_proba(X_test)[:,1]
        e=self.evaluation(target,y_test,y_pred,grid,X_test,X_train,y_train,models,y_p)
        self.evaluation2(y_test,y_pred,grid,X_test,X_train,y_train)
        return(e)


    def neural_class(self,target,X_train,y_train,X_test,y_test,alpha=0.5,learning_rate_init=0.005,max_iter=400,hidden_layer_sizes=(400,)):
        models="nn"
        scaler=MaxAbsScaler()
        mlpc=MLPClassifier(solver="adam",verbose=True, alpha=alpha, batch_size="auto", learning_rate="adaptive", learning_rate_init=learning_rate_init, early_stopping=True,max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes)
        pipe=Pipeline(steps=[("scaling",scaler),("neural",mlpc)])
        pipe.fit(X_train, y_train) 
        pkl_filename = "workspace/test_user/qsar_models/%s_nn_model.joblib"%target
        with open(pkl_filename, 'wb') as file:
            joblib.dump(pipe, file)
        y_pred = pipe.predict(X_test)
        y_p = pipe.predict_proba(X_test)[:,1]
        e=self.evaluation(target,y_test,y_pred,pipe,X_test,X_train,y_train,models,y_p)
        self.evaluation2(y_test,y_pred,pipe,X_test,X_train,y_train)
        print(pkl_filename)
        return(e)


    #SVM classifier
    def svm_class(self,target,X_train,y_train,X_test,y_test,C=1):
        models="svm"
        scaler=MaxAbsScaler()
        #SVM with tuned parameters
        svc=LinearSVC(penalty="l2",C=C,class_weight="balanced",verbose=True,max_iter=15000)
        svc = CalibratedClassifierCV(svc) 
        #lin_svc = svm.LinearSVC().fit(X_train, y_train)
        grid=Pipeline(steps=[("scaling",scaler),("svc",svc)])


      #  best_pipe = grid.best_estimator_ 
      #  print(best_pipe)

        grid.fit(X_train, y_train)
        pkl_filename = "workspace/test_user/qsar_models/"+target+"_svm_model.joblib"
        with open(pkl_filename, 'wb') as file:
            joblib.dump(grid, file)
        y_pred = grid.predict(X_test)
        y_p = grid.predict_proba(X_test)[:,1]
        e=self.evaluation(target,y_test,y_pred,grid,X_test,X_train,y_train,models,y_p)
        self.evaluation2(y_test,y_pred,grid,X_test,X_train,y_train)
        return(e)

    # logistic regression
    def logit_class(self,target,X_train,y_train,X_test,y_test,C=1,solver="liblinear"):
        models="lr"
        scaler=MaxAbsScaler()
        lg_classifier = LogisticRegression(random_state=0,class_weight="balanced",verbose=1,max_iter=1000,C=C,solver=solver)
        grid=Pipeline(steps=[("scaling",scaler),("logit",lg_classifier)]) 
        grid.fit(X_train, y_train)
        pkl_filename = "workspace/test_user/qsar_models/"+target+"_lr_model.joblib"
        with open(pkl_filename, 'wb') as file:
            joblib.dump(grid, file)
        y_pred = grid.predict(X_test)
        y_p = grid.predict_proba(X_test)[:,1]
        e=self.evaluation(target,y_test,y_pred,grid,X_test,X_train,y_train,models,y_p)
        self.evaluation2(y_test,y_pred,grid,X_test,X_train,y_train)
        return(e)

    def gbdt_class(self,target,X_train,y_train,X_test,y_test,max_depth=5,min_samples_leaf=4,subsample=0.8,n_estimators=500):
        models="gbdt"
        auc_scorer=metrics.make_scorer(metrics.roc_auc_score)
        grid=GradientBoostingClassifier(max_depth=max_depth,random_state=45,verbose=1,min_samples_leaf=min_samples_leaf,subsample=subsample,n_estimators=n_estimators) 
        grid.fit(X_train, y_train)
        pkl_filename = "workspace/test_user/qsar_models/"+target+"_gbdt_model.joblib"
        with open(pkl_filename, 'wb') as file:
            joblib.dump(grid, file)
        y_pred = grid.predict(X_test)
        y_p = grid.predict_proba(X_test)[:,1]
        e=self.evaluation(target,y_test,y_pred,grid,X_test,X_train,y_train,models,y_p)
        self.evaluation2(y_test,y_pred,grid,X_test,X_train,y_train)
        return(e)


        
   
    def trainer(self,target,model,**kwargs): #input is the string formed name of the file(name of cell/protein)
        #read data from database
       #select labels with value "1" and "0"
        filename="%s.csv"%target


        df=pd.read_csv(filename)
        df=df[df["target"]!=99]
        X = df.iloc[:,3:-1]
        y = df.iloc[:,-1] 
        y=pd.to_numeric(y)
        Cnames=[]
        for i in range(0,1049):
            Cnames.append(str(i))  
        json_doc=[]
        X=csr_matrix(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=2)
        print(type(y_train[2]))

        precision_scorer=metrics.make_scorer(metrics.precision_score)
        if model=="svm":
            try:
                svmj=self.svm_class(target,X_train,y_train,X_test,y_test,**kwargs)
                json_doc.append(svmj)
            except:
                logger.exception("error in training %s  model"%model)
        if model=="lr":
            try:
                lrj=self.logit_class(target,X_train,y_train,X_test,y_test,**kwargs)
                json_doc.append(lrj)
            except:
                logger.exception("error in training %s  model"%model)
        if model=="gbdt":
            try:
                xgbtj=self.gbdt_class(target,X_train,y_train,X_test,y_test,**kwargs)
                json_doc.append(xgbtj)
            except:
                logger.exception("error in training %s  model"%model)
        if model=="rf":
            try:
                rfj=self.randomforest_class(target,X_train,y_train,X_test,y_test,**kwargs)
                json_doc.append(rfj)
            except:
                logger.exception("error in training %s  model"%model)
        if model=="nn":
            try:
                nnj=self.neural_class(target,X_train,y_train,X_test,y_test,**kwargs)
                json_doc.append(nnj)
            except:
                logger.exception("error in training %s  model"%model)
        import json
       
        with open("workspace/test_user/qsar_models/%s_%s_evaluation"%(target,model), 'w') as fout:
            json.dump(json_doc, fout)
            fout.write("\n")
            logger.info("evaluation results saved to:%s_model_evaluation"%target)
            logger.info("model saved to:","qsar_models/%s_%s_model.joblib"%(target,model))

#    def model_training(self,target,model):
#        
#    
#        #tr=train("chembl2026")
#        #kwargs={"C":0.5}
#        #tr.trainer("svm",**kwargs)
#        #kwargs={"alpha":0.5,"hidden_layer_sizes":(300,)}
#        #tr.trainer("nn",**kwargs)
#        #
#        #kwargs={"n_estimators":200,"max_depth":4}
#        #tr.trainer("rf",**kwargs)
#        #
#        kwargs={"n_estimators":200,"max_depth":7}
#        tr.trainer("gbdt",**kwargs)
#
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,format='[%(asctime)s] [%(name)s:%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)s] %(message)s',datefmt='')
