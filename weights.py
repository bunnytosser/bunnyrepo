import pandas as pd
import numpy as np
import math
import os
def entropy_eval(filepath, cols,skiprows=0): #熵值法
    if type(filepath)==str and filepath[-5:]=='.xlsx':
        df1 = pd.read_excel(filepath, index_col=0,skiprows=skiprows)
        if os.path.exists(filepath) == False:
               fileerror="文件名不存在！请检查后重新调用函数。"
               print(fileerror)
               return False
    elif type(filepath)==str and filepath[-4:]=='.csv':
        df1 = pd.read_csv(filepath, index_col=0,skiprows=skiprows)
        if os.path.exists(filepath) == False:
               fileerror="文件名不存在！请检查后重新调用函数。"
               print(fileerror)
               return False
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
        print(weights)
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



def stdcal(filepath,cols,skiprows=0): #变异系数法
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
        df1 = pd.read_excel(filepath, index_col=0,skiprows=skiprows)
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
    return{'变异系数':a,'权重':b}

def opt_weights_calculator(filepath,cols,skiprows=0):
    if entropy_eval(filepath, cols,skiprows)==False or stdcal(filepath,cols,skiprows)==False:
        return False
    entrophy_weights=np.array(list(entropy_eval(filepath, cols,skiprows).values()))
    variation_weights=stdcal(filepath,cols,skiprows)['权重']
    std_entrophy=np.std(entrophy_weights)
    std_var=np.std(variation_weights)
    if std_entrophy>std_var:
        opt_weights=variation_weights
        opt_method='变异系数法'
    elif std_entrophy<std_var:
        opt_weights=entrophy_weights
        opt_method='熵值法'
    else:
        opt_weights=entrophy_weights
        opt_method='熵值法或者变异系数法'
    return{'最佳权重':opt_weights,'方法':opt_method}

##file:'C:\\Users\\USUARIO\\Desktop\\work\\timeseries.xlsx'
#['出账用户数（万）','流量单价（元/M）']
##test:opt_weights_calculator('C:\\Users\\USUARIO\\Desktop\\work\\timeseries.xlsx',['出账用户数（万）','流量单价（元/M）'],1)