# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:11:34 2017

输入数据框，或者excel文件，对指定的列（默认全部列）进行空值处理。该函数的主要特性：
1. 自动对输入的数据进行时间序列化转化处理，输出为已经进行了空值处理的dict，dict的每一个元素为
每一列的时间序列。
2. 用户可以指定列进行处理，列可以是数字的list，也可以是具体列名的list，并自动识别检测不存在的列。默认
可以自动处理全部列。
3. 用户可选三种空值处理方式：常数填充（method取数字），空值移除（‘omit’，为默认选项），或者前后均值处理（method
为‘average’）.前后均值为将空值出现的前后最邻近的非空值取平均数。若其之前或之后没有非空值，则取值等于其最
临近的非空值。
4.返回结果为一个dict。其每个元素取值为每列的时间序列。
5. 用户必须要输入事件序列的起止时间，和频率。例如：
ts_nullhandling(r'C:\Users\USUARIO\Documents\Tencent Files\3215313729\FileRecv\序列预测样本数据2.xlsx','2011-6','2014-6','m',method='average')

ts_nullhandling(r'C:\Users\USUARIO\Documents\Tencent Files\3215313729\FileRecv\序列预测样本数据2.xlsx','2011-6','2014-6','m',cols=[3,4,5]),method=[9,10,11])

@author: Jia
"""


#####
import pandas as pd
import numpy as np
from datetime import datetime
import os

def ts_nullhandling(filepath,start,end,freq,cols=0,method='omit'): 
    if type(filepath)==str and filepath[-5:]=='.xlsx':
        if os.path.exists(filepath) == False:
               fileerror="文件名不存在！请检查后重新调用函数。"
               print(fileerror)
               return False
        df1 = pd.read_excel(filepath)
    elif type(filepath)==str and filepath[-4:]=='.csv':
        if os.path.exists(filepath) == False:
               fileerror="文件名不存在！请检查后重新调用函数。"
               print(fileerror)
               return False
        df1 = pd.read_csv(filepath)
    elif type(filepath)==pd.core.frame.DataFrame or type(filepath)==pandas.core.series.Series:
        df1=filepath
    else:
        print('不支持的文件扩展名或者不存在的数据框')
        return False
        #####
    index=pd.date_range(start,end,freq=freq)
    ##转化为时间序列数据框
    df1=pd.DataFrame(df1[:len(index)].values,index=index,columns=df1.columns)
    #调整选取的列名
    if cols==0:
        cols=list(df1.columns)
    elif type(cols[0])==str and set(np.array(cols))-set(np.array(df1.dtypes.index))!=set() :
        wrongcol=set(np.array(cols))-set(np.array(df1.dtypes.index))
        print('错误的列',wrongcol)
        return False
    elif type(cols[0])==int:
        if set(cols)-set(np.arange(len(df1.columns)))!=set():
            wrongcolumn=set(cols)-set(np.arange(len(df1.columns)))
            print('错误的列',wrongcolumn)
            return False
        else:
            cols=list(df1.columns[cols])
    newseries={}
    c=0
    for i in cols:
        t1=np.isnan(df1[i])
        if method=='omit':
            newseries[i]=df1[i][~t1]
        elif method=='average':
            for j in np.arange(len(df1[i])):
                if np.isnan(df1[i][j])==True:
#                    if i==0:
#                        df1[i][j]=next(x for x in df1[i][j:] if np.isnan(x)==False)
#                    elif i==len(df1[i]):
#                        df1[i][j]=next(x for x in df1[i][:j].ix[::-1] if np.isnan(x)==False)
#                    else:
                        try:
                            nex=next(x for x in df1[i][j:] if np.isnan(x)==False)
                        except:
                            nex=next(x for x in df1[i][:j].ix[::-1] if np.isnan(x)==False)
                        try:
                            las=next(x for x in df1[i][:j].ix[::-1] if np.isnan(x)==False)
                        except:
                            las=next(x for x in df1[i][j:] if np.isnan(x)==False)
                        df1[i][j]=(las+nex)/2
            newseries[i]=df1[i]
        else:
             df1[i][t1]=method[c]
             newseries[i]=df1[i]
             c=c+1
    return(newseries)
        
        
            
# try:
#    nex=next(x for x in df1['流量单价'][3:] if np.isnan(x)==False)
#    print(nex)
#except:
#    las=next(x for x in df1['流量单价'][:-3].ix[::-1] if np.isnan(x)==False)
#    print(las)

##exp_smoothing(r'C:\Users\USUARIO\Documents\Tencent Files\3215313729\FileRecv\序列预测样本数据2.xlsx','2011-6','2014-6','m',method='average')
##
#df1 = pd.read_excel(r'C:\Users\USUARIO\Documents\Tencent Files\3215313729\FileRecv\序列预测样本数据2.xlsx')
#index=pd.date_range('2011-06','2014-06',freq='m')
#df1=pd.DataFrame(df1[:len(index)].values,index=index,columns=df1.columns)
#cols=list(df1.columns)
#ts=pd.DataFrame(series.values,index=index,columns=series.columns)
#
#ts1=ts.ix[:,2]
#
##方法1：舍弃
#ts1=ts1[ts1.notnull()]
##方法2：常熟填充
#ts1=ts1.fillna(0)
##方法3：均值填充
#for i in np.arange(len(ts1)):
#    if ts1[i] is None:
#        if i==0:
#            ts1[i]=ts1[i+1]
#        elif i==len(ts1):
#            t11[i]=ts1[i-1]
#        else:
#            ts1[i]=(ts1[i-1]+ts1[i+1])/2
