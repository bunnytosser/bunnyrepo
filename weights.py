# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:30:58 2017

@author: USUARIO
"""

import pandas as pd
import numpy as np
import math
import os
from sklearn.decomposition import PCA
def entropy_w(filepath, cols,skiprows=0): #熵值法
    if type(filepath)==str and filepath[-5:]=='.xlsx':
        if os.path.exists(filepath) == False:
               fileerror="文件名不存在！请检查后重新调用函数。"
               print(fileerror)
               return False
        df1 = pd.read_excel(filepath, index_col=0,skiprows=skiprows)
    elif type(filepath)==str and filepath[-4:]=='.csv':
        if os.path.exists(filepath) == False:
               fileerror="文件名不存在！请检查后重新调用函数。"
               print(fileerror)
               return False
        df1 = pd.read_csv(filepath, index_col=0,skiprows=skiprows)
    elif type(filepath)==pd.core.frame.DataFrame:
        df1=filepath
    else:
        print('不支持的文件扩展名或者不存在的数据框')
        return False
    # df1 = pd.read_excel('D:\work_bonc\Python_WorkSpace\weights\samples.xlsx', index_col=0)
    if set(cols)<=set(np.arange(len(df1.columns)))or set(np.array(cols)) <=set(np.array(df1.dtypes.index)) :
        df2 = df1.copy()# 信息熵字典
        entropy = {}# 差异系数字典
        diversity_factor = {}# 权重字典
        weights = {}

        def plnp(s):
            if s == 0:
                return np.nan  # 如果这个地方不处理的话，会报错ValueError: math domain error
            else:
                return -1*s*math.log(s, math.e)

        for columns in cols:
            df2[columns + '_比重'] = df2[columns] / df2[columns].sum()
            df2[columns + '_-plnp'] = df2[columns + '_比重'].apply(plnp)
            entropy[columns + '_信息熵'] = (1/math.log(len(df2[columns]), math.e) * df2[columns + '_-plnp'].sum())
            diversity_factor[columns + '_差异系数'] = 1 - entropy[columns + '_信息熵']

        for columns in cols:
            weights[columns + '_权重'] = diversity_factor[columns + '_差异系数'] / sum(diversity_factor.values())
        return weights

# entropy_evaluation('samples002.xlsx', ['旅游总收入增长率', '旅游成本回收率'])
# entropy_evaluation('samples001.xlsx', ['体育', '数学'])
    elif type(cols[0])==int:
        wrongcolumn=set(cols)-set(np.arange(len(df1.columns)))
        print('错误的列',wrongcolumn)
        return False
    else:
        wrongcol=set(np.array(cols))-set(np.array(df1.dtypes.index))
        print('错误的列',wrongcol)
        return False



def std_w(filepath,cols,skiprows=0): #变异系数法
    ###type设定输入的colname list的元素类型，是字符型还是数值型，默认是数字型为0
    a=[]
    weights=[]
    if type(filepath)==str and filepath[-5:]=='.xlsx':
        if os.path.exists(filepath) == False:
               fileerror="文件名不存在！请检查后重新调用函数。"
               print(fileerror)
               return False
        df1 = pd.read_excel(filepath, index_col=0,skiprows=skiprows)
    elif type(filepath)==str and filepath[-4:]=='.csv':
        if os.path.exists(filepath) == False:
               fileerror="文件名不存在！请检查后重新调用函数。"
               print(fileerror)
               return False
        df1 = pd.read_csv(filepath, index_col=0,skiprows=skiprows)
    elif type(filepath)==pd.core.frame.DataFrame:
        df1=filepath
    else:
        print('不支持的文件扩展名或者不存在的数据框')
        return False
    data=df1
    if type(cols[0])==int:
        if set(cols)<=set(np.arange(len(data.columns)))or set(np.array(cols)) <=set(np.array(data.dtypes.index)) :   
          for value in cols:
            valuestd=np.std(data.ix[:,value])
            valueavg=np.average(data.ix[:,value])
            a.append(valuestd/valueavg)
        else:
          wrongcolumn=set(cols)-set(np.arange(len(data.columns)))
          print('错误的列',wrongcolumn)
          return False
    else:
        if set(cols)<=set(np.arange(len(data.columns)))or set(np.array(cols)) <=set(np.array(data.dtypes.index)) :   
          for value in cols:
            valuestd=np.std(data[value])
            valueavg=np.average(data[value])
            a.append(valuestd/valueavg)
        else:
            wrongcol=set(np.array(cols))-set(np.array(data.dtypes.index))
            print('错误的列',wrongcol)
            return False
    sumcoef=np.sum(a)
    sumlist=[sumcoef]*len(a)
    b=np.array(a)/np.array(sumlist)
    stdweights={}
    j=0
    for i in cols:
        stdweights[i]=b[j]
        j=j+1    
    return stdweights

def PCA_w(filepath, cols,skiprows=0):
    if type(filepath)==str and filepath[-5:]=='.xlsx':
        if os.path.exists(filepath) == False:
               fileerror="文件名不存在！请检查后重新调用函数。"
               print(fileerror)
               return False
        df1 = pd.read_excel(filepath, index_col=0,skiprows=skiprows)
    elif type(filepath)==str and filepath[-4:]=='.csv':
        if os.path.exists(filepath) == False:
               fileerror="文件名不存在！请检查后重新调用函数。"
               print(fileerror)
               return False
        df1 = pd.read_csv(filepath, index_col=0,skiprows=skiprows)
    elif type(filepath)==pd.core.frame.DataFrame:
        df1=filepath
    else:
        print('不支持的文件扩展名或者不存在的数据框')
        return False
    data = df1
    if (set(cols) - set(data.columns)) != set():
        print('以下列名称不存在: %s' % (set(cols) - set(data.columns)))
        return False
    else:
        data = data.loc[:, cols]
        # 将数据转化为numpy.ndarry,方便后续的计算处理
        arr = np.array(data).astype(np.float)

        # 将数据使用PCA处理
        pca = PCA(n_components=len(data.columns))
        # 将arr使用PCA之后转换为new_arr
        new_arr = pca.fit_transform(arr)
        # 特征根(是否已经降序排序，需要查资料确认？，在后面暂时手动降序排序处理了)
        eigVals = pca.explained_variance_

        # 特征向量(行为特征向量，已单位化)
        eigVects = pca.components_

        # 将特征值从大到小进行排序，返回排序后的索引值
        eigValIndex = np.argsort(-eigVals)

        # 特征值对应的方差占比
        eig_var_ratio = pca.explained_variance_ratio_

        # 选择方差贡献率大约全部90%的前K个特征值以及对应的特征向量
        final_index = []
        tmp_sum = 0
        total_sum = sum(eig_var_ratio)
        for index in eigValIndex:
            final_index.append(index)
            tmp_sum += eig_var_ratio[index]
            if tmp_sum >= 0.9:
                break
        final_eigVals = []
        final_eigVects = []
        final_eigVarRatio = []
        for index in final_index:
            final_eigVals.append(eigVals[index])
            final_eigVects.append(eigVects[index])
            final_eigVarRatio.append(eig_var_ratio[index])

        # 指标在不同主成分线性组合中的系数：载荷数/特征根的开方（载荷数为特征向量对应的元素）
        coefficient_1 = []
        for i in range(len(final_eigVects)):
            coefficient_1.append(final_eigVects[i] / math.sqrt(final_eigVals[i]))


        # 指标的方差贡献率：以各个主成分的方差贡献率为权重，对线性组合中的系数做加权平均
        coefficient_1 = np.array(coefficient_1).astype(np.float)  # 将list转化为numpy.ndarray
        coefficient_2 = np.dot(np.mat(final_eigVarRatio), coefficient_1) / sum(final_eigVarRatio)

        #指标权重归一化
        coefficient_3 = coefficient_2 / np.sum(coefficient_2)
        coefficient_3 = np.array(coefficient_3)

        weights = {}
        for i in range(len(data.columns)):
            weights[data.columns[i]] = coefficient_3[0][i]
        return weights

def opt_weights(filepath,cols,skiprows=0):
    if entropy_w(filepath, cols,skiprows)==False or std_w(filepath,cols,skiprows)==False or PCA_w(filepath, cols,skiprows)==False:
        return False
    entropy_weights=np.array(list(entropy_w(filepath, cols,skiprows).values()))
    variation_weights=np.array(list(std_w(filepath, cols,skiprows).values()))
    pcaweights=np.array(list(PCA_w(filepath, cols,skiprows).values()))
    weights_list=[0,0,0]
    weights_list[0]=np.std(entropy_weights)
    weights_list[1]=np.std(variation_weights)
    weights_list[2]=np.std(pcaweights)
    if min(weights_list)==weights_list[1]:
        opt_weights=variation_weights
        opt_method='变异系数法'
    elif min(weights_list)==weights_list[0]:
        opt_weights=entropy_weights
        opt_method='熵值法'
    elif min(weights_list)==weights_list[2]:
        opt_weights=pcaweights
        opt_method='主成分分析法'
    return{'熵值法权重结果':entropy_w(filepath, cols,skiprows),'变异系数法权重结果':std_w(filepath, cols,skiprows),'主成分分析法权重结果':PCA_w(filepath, cols,skiprows),'最优方法':opt_method}

##file:'C:\\Users\\USUARIO\\Desktop\\work\\timeseries.xlsx'
#['出账用户数（万）','流量单价（元/M）']
##test:opt_weights_calculator('C:\\Users\\USUARIO\\Desktop\\work\\timeseries.xlsx',['出账用户数（万）','流量单价（元/M）'],1)

##file:'C:\\Users\\USUARIO\\Desktop\\work\\timeseries.xlsx'
#['出账用户数（万）','流量单价（元/M）']
##test:opt_weights('C:\\Users\\USUARIO\\Desktop\\work\\timeseries.xlsx',['出账用户数（万）','流量单价（元/M）','流量收入（出账万）'],1)
