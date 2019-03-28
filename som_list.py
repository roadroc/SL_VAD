#MiniSom聚类将数据集替换为聚类数据集，仅由多种聚类结果按原顺序构成，因此需要拆分数据集，构建列表
import numpy
from little_tools import data_unique

def som_list(data,label,som_data):#输入数据集和聚类数据集，输出每个类簇对应的数据列表及标签列表
    unique_data=data_unique(som_data)#删除重复行后，二维数组仅由类簇构成
    s_list=[]#构建类簇数据列表
    l_list=[]#构建类簇数据对应的标签列表
    for i in unique_data:
        temp=[]#构建临时列表
        temp_label=[]
        for a,b in enumerate(som_data):
            if b==i:
                temp.append(data[a])
                temp_label.append([label[a]])
        s_list.append(temp)
        l_list.append(temp_label)
    return s_list,l_list
