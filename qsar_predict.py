#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np
import pdb
import os
import sys
import copy
import pandas as pd
from scipy import stats
from scipy.stats import norm
from ast import literal_eval

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from collections import defaultdict
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.externals import joblib
import mysql.connector as sql
import json

import logging
import logging.config

logger = logging.getLogger("LOG")

class prediction:
    def __init__(self):
        pass

    def predict(self, feature_df, model, compound_file):
        #df=pd.DataFrame(ll)
        results_file = "../results_data/%s_%s_%s" % (target, model, compound_file)
        #pdb.set_trace()
       # filename=target+".csv"
       # feature_df=pd.read_csv(filename)
        df=feature_df.iloc[:,1:]
       # print(df.columns)
        smiles=feature_df.iloc[:,0]
        logger.info(df.shape)
        jl_filename="../qsar_models/" + target + "_" + model +"_model.joblib"
        logger.info(jl_filename)
        import os.path
        if os.path.isfile(jl_filename) is True:
            with open(jl_filename, 'rb') as file:
                models=joblib.load(file)

                y=models.predict_proba(df)[:,1]
                y=pd.Series(y)
                final=pd.concat([smiles,y],axis=1)
                logger.info(final.columns)
                final["label"]=np.where(final[0]>0.7,"high", np.where(final[0]<0.4,"low","medium"))
                logger.info(final[190:200])      
                final.to_csv(results_file)
                return True
        else:
            return False
                              
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
            format='[%(asctime)s] [%(name)s:%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)s] %(message)s',
                    datefmt='')
