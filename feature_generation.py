# -*- coding: utf-8 -*-
"""
Jia 13/12/2018

env: python2.7
"""

from pydpi import pydrug
from pydpi.drug import constitution,getmol,topology,kappa,charge,fingerprint
from pydpi.protein import getpdb
from pydpi import pypro
from pydpi.drug import connectivity
from pydpi.pydrug import Chem
from pydpi.drug import topology
from pydpi.pypro import PyPro
from pydpi.drug import kappa
from pydpi.drug import molproperty
import numpy as np
import itertools
from pydpi import pydpi
import pandas as pd
from pandas import DataFrame
import logging
import logging.config
import os.path
import os
logger = logging.getLogger("LOG")
drug=pydrug.PyDrug()
class get_features:
    def __init__(self):
        pass
    def uniprot_converter(self,ps,ido,L2):
        try:
            dpi=pydpi.PyDPI()
    #         print prot
    #         ps=dpi.GetProteinSequenceFromID(prot)
    #         print ps
            dpi.ReadProteinSequence(ps)
            protein=PyPro()
            protein.ReadProteinSequence(ps)
            allp=protein.GetALL()
            allp["ID"]=ido
            L2.append(allp)
      #      print(smi)
        except:
            logger.info("Unable to convert the sequence into features")
            return False
    


    def smiles_converter(self,smi,L):
        res=[]
        try:
            mol=Chem.MolFromSmiles(smi)
            res.append(constitution.GetConstitutional(mol))
            res.append(connectivity.GetConnectivity(mol))
            res.append(fingerprint.CalculateMACCSFingerprint(mol)[1])
            try:
                res.append(kappa.GetKappa(mol))
            except:
                pass
            try:
                drug.ReadMolFromSmile(smi)
                res.append(drug.GetMOE())
            except:
                pass
            try:
                res.append(charge.GetCharge(mol))
            except:
                pass
        except:
            pass
        super_dict = {}  # uses set to avoid duplicates
        if len(res)>=1:
            for d in res:
                for k, v in d.items():  # use d.iteritems() in python 2
                    super_dict[k]=v
        super_dict["smiles"]=smi.strip()
        L.append(super_dict)

###convert proteins

    def file_len(self,fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    def protein_features(self,prot_in,prot_out):
        L2=[]
        size = self.file_len(prot_in)
        f=open(prot_in)
        sequences=[]
        ids=[]
        seqfull=""
        for i,line in enumerate(f):
            if line[0]==">":
                if len(line[1:].strip())>=10 and "|" in line:
                    idi=line.split("|")
                    idi=idi[1]
                    ids.append(idi)
                else:
                    ids.append(line[1:].strip())
                if seqfull!="":
                    sequences.append(seqfull)
                seqfull=""
                continue
            elif line[0]==" ":
                continue
            else:
                seq=line.strip()
                seqfull+=seq
                if i==size-1:
                    sequences.append(seqfull)
        for seq,ido in zip(sequences[:2],ids[:2]):
            self.uniprot_converter(seq,ido,L2)
        cols=[]
        for c in L2:
            for i in c.keys():
                if i not in cols:
                    cols.append(i)
        DF=pd.DataFrame.from_records (L2,columns=cols)
        # DF.to_csv("/Users/xave/Downloads/CPI-1022/smiles_pydpi2.csv",sep="\t")
        ID = DF['ID']
        DF.drop(labels=['ID'], axis=1,inplace = True)
        DF.insert(0, 'Uniprot ID', ID)
        DF.to_csv(prot_out)
        logger.info("%s sequence(s) converted successfully"%len(DF))

    def ligand_features(self,smi_in,smi_out):
        L=[]
        fs=open(smi_in)
        smiles=[]
        for i in fs:
            smiles.append(i)
        for smi in smiles:
            self.smiles_converter(smi,L)
        cols=[]
        for c in L:
            for i in c.keys():
                if i not in cols:
                    cols.append(i)
        DF=pd.DataFrame.from_records (L,columns=cols)
        # DF.to_csv("/Users/xave/Downloads/CPI-1022/smiles_pydpi2.csv",sep="\t")
        ID = DF['smiles']
        DF.drop(labels=['smiles'], axis=1,inplace = True)
        DF.insert(0, 'smiles', ID)
        DF.to_csv(smi_out)
        logger.info("%s SMILES code(s) converted successfully"%len(DF))



