#小工具集合
import sklearn
import numpy

def Z_Score(data):#输入二维数组，标准化处理数据（所有数据都聚集在0附近，方差为1）
    z_data=sklearn.preprocessing.scale(data, axis=1)
    return z_data

def sort_data(data):#输入二维数组，升序排列每行数据
    s_data=numpy.sort(data)
    return s_data

def sorted_data(data):#输入二维数组，升序排列每行数据
    data=data.tolist()#二维数组转为二维列表
    sed_data=[]
    for i in data:
        i=i.sorted(data,reverse=True)
        sed_data.append(i)

    sed_data=numpy.array(sed_data)
    return sed_data

def data_unique(data):#输入二维数组，去除重复行
    unique_data=numpy.unique(data)
    return unique_data

def one_hot(x,y):#输入SOM拓扑网络(x,y)，设计与之对应的独热编码
    z=x*y
    one_hot_label=numpy.zeros(z,z)
    for i in one_hot_label:
        one_hot_label[i]=1
    return one_hot_label