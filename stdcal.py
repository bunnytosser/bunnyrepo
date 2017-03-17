# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:54:15 2017

@author: USUARIO
"""

import pandas as pd
import numpy as np
def stdcal(colname,data):
    ###type设定输入的colname list的元素类型，是字符型还是数值型，默认是数字型为0
    a=[]
    weights=[]
    if set(colname)<=set(np.arange(len(data.columns)))or set(np.array(colname)) <=set(np.array(data.dtypes.index)) :   
         for value in colname:
            valuestd=np.std(data.ix[value])
            valueavg=np.average(data.ix[value])
            a.append(valuestd/valueavg)
    elif type(colname[0])==int:
        wrongcolumn=set(colname)-set(np.arange(len(data.columns)))
        return{'错误的列':wrongcolumn}
    else:
        wrongcol=set(np.array(colname))-set(np.array(data.dtypes.index))
        return{'错误的列':wrongcol}
    sumcoef=np.sum(a)
    sumlist=[sumcoef]*len(a)
    b=np.array(a)/np.array(sumlist)
    return{'变异系数':a,'权重':b}
