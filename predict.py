import numpy as np
import pdb
import os
import sys
import copy
import pandas as pd
from scipy import stats
from scipy.stats import norm
# from ast import literal_eval

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import preprocessing

from sklearn import metrics


from sklearn.pipeline import Pipeline
from sklearn import svm
from collections import defaultdict
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.externals import joblib

import json
from rdkit import Chem
from rdkit.Chem import PandasTools

import logging
import logging.config
import os.path
logger = logging.getLogger("LOG")
os.chdir("/Users/xave/Downloads/cpi_api/")
# results_fileb="results/results_binding.csv"
# results_filea="results/results_affinity.csv"
class prediction:
    def __init__(self):
        pass
    def preprocessing(self,prot_in,lig_in):
        prots=pd.read_csv("out_prot.csv")
        ligs=pd.read_csv("out_smi.csv")
        with open("features.txt") as f:
            prots=pd.read_csv("out_prot.csv")
            ligs=pd.read_csv("out_smi.csv")
            feat=f.readlines()
            feat_list = [x.strip() for x in feat]
            p_feat=feat_list[:1469]
            l_feat=feat_list[1469:]
            p_data=pd.DataFrame(0,index=prots["Uniprot ID"],columns=p_feat)
            
            l_data=pd.DataFrame(0,index=ligs["smiles"],columns=l_feat)
            for p in p_data.columns:
                pname=p
                if p[-2:]=="_x":
                    pname=p[:-2]
                if pname in prots.columns:
                    p_data[p]=prots[pname].tolist()
            for l in l_data.columns:
                lname=l
                if l[-2:]=="_y":
                    lname=l[:-2]
                if lname in ligs.columns:
                    l_data[l]=ligs[lname].tolist()
                    
        p_data=p_data.iloc[:2,:]
        p_data['tmp'] = 1
        p_data["Uniprot ID"]=p_data.index
        l_data['tmp'] = 1
        l_data["smiles"]=l_data.index
        DF=pd.merge(p_data,l_data,how="outer",on=['tmp'])
        DF=DF.drop("tmp",axis=1)
        ID = DF['Uniprot ID']
        smiles=DF["smiles"]
        DF.drop(labels=['Uniprot ID',"smiles"], axis=1,inplace = True)
        DF.insert(0,'UniProtID', ID)
        DF.insert(0,"smiles",smiles)
        DF=DF.replace([np.inf, -np.inf], np.nan)
        DF=DF.fillna(0)
        return DF


#     def affinity(self, prot_in,lig_in,outpath="results/results_affinity.csv"):
# #         os.chdir("/Users/xave/Downloads/cpi_api/")
        
#         DF=self.preprocessing(prot_in,lig_in)
#         X=DF.iloc[:,2:]
#         print (DF.columns)
#         logger.info(X.shape)
#         jl_filename = "models/gbdt_regression.joblib"
#         logger.info(jl_filename)
#         if os.path.isfile(jl_filename) is True:
#             with open(jl_filename, 'rb') as file:
#                 models=joblib.load(file)
#                 y=pd.Series(models.predict(X))
#                 y=y.rename("predicted_affinity")
#                 smiles=DF["smiles"]
#                 prot=DF["UniProtID"]
#                 final=pd.concat([smiles,prot,y],axis=1)
#                 logger.info(final.columns)
#                 logger.info(final[0:10])   
#                 final.to_csv(outpath)
#                 pp_out = "results/affinity_out.sdf"
# # PandasTools.AddMoleculeColumnToFrame(pp,0)
#                 PandasTools.AddMoleculeColumnToFrame(final,'smiles','Molecule')
#                 PandasTools.WriteSDF(final, pp_out, molColName='Molecule',properties=list(final.columns))
#                 return True
#         else:
#              return False
#     def binding(self, prot_in,lig_in,outpath="results/results_binding.csv"):
# #         os.chdir("/Users/xave/Downloads/cpi_api/")
#         DF=self.preprocessing(prot_in,lig_in)
#         X=DF.iloc[:,2:]
#         print (X.columns)
#         logger.info(X.shape)
#         cl_filename = "models/gbdt_model.joblib"
#         logger.info(cl_filename)
#         if os.path.isfile(cl_filename) is True:
#             with open(cl_filename, 'rb') as file:
#                 models=joblib.load(file)
#                 y=pd.Series(models.predict_proba(X)[:,1])
#                 y=y.rename("probability")
#                 smiles=DF["smiles"]
#                 prot=DF["UniProtID"]
#                 final=pd.concat([smiles,prot,y],axis=1)
#                 final["predicted_label"]=np.where(final.probability>0.7,"high", np.where(final.probability<0.4,"low","medium"))
#                 logger.info(final.columns)
#                 logger.info(final[0:10])
#                 final.to_csv(outpath)
#                 return True
#         else:
#             return False
#             def affinity(self, prot_in,lig_in,outpath="results/results_affinity.csv"):
# #         os.chdir("/Users/xave/Downloads/cpi_api/")
    def binding_affinity(self, prot_in,lig_in,outpath="results/results_affinity_binding.csv"):
        DF=self.preprocessing(prot_in,lig_in)
        X=DF.iloc[:,2:]
        print (DF.columns)
        logger.info(X.shape)
        jl_filename = "models/gbdt_regression.joblib"
        cl_filename = "models/gbdt_model.joblib"
        if os.path.isfile(jl_filename) is True:
            with open(jl_filename, 'rb') as file:
                models=joblib.load(file)
                y=pd.Series(models.predict(X))
                ya=y.rename("predicted_affinity")
        else:
             logger.info("no model available")
        if os.path.isfile(cl_filename) is True:
            with open(cl_filename, 'rb') as file:
                models=joblib.load(file)
                yb=pd.Series(models.predict_proba(X)[:,1])
        else:
            logger.info("no model available")
        smiles=DF["smiles"]
        prot=DF["UniProtID"]
        final=pd.concat([smiles,prot,ya,yb],axis=1)
        final.columns=["smiles","Uniprot ID","affinity","probability"]
        final["predicted_label"]=np.where(final.probability>0.7,"high", np.where(final.probability<0.4,"low","medium"))
        logger.info(final.columns)
        logger.info(final.columns)
        logger.info(final[0:10])   
        final.to_csv(outpath)
        pp_out = "results/affinity_out.sdf"
        PandasTools.AddMoleculeColumnToFrame(final,'smiles','Molecule')
        PandasTools.WriteSDF(final, pp_out, molColName='Molecule',properties=list(final.columns))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
            format='[%(asctime)s] [%(name)s:%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)s] %(message)s',
                    datefmt='')
