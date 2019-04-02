#工具类
#涉及到的数据均为numpy格式
import sklearn,scipy,numpy
from minisom import MiniSom

class Main_Tools(object):#主要的工具，负责初始阶段特征数据集和聚类数据集的获取
    def __init__(self, path1, path2, x, y, sigma, learning_rate, epoch):#为了保持数据的统一，采用的数据集为Kaldi提供的MFCC特征数据集，特征参数量为20，输入其路径，且mat文件只存在一个变量mfcc
        self.path1 = path1
        self.path2 = path2
        self.x = x#SOM结构
        self.y = y
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.epoch = epoch#SOM迭代次数

    def get_mfcc(self):#获取mat文件内的MFCC特征数据集
        mfcc_dict = scipy.io.loadmat(self.path1)#从路径导入mat文件，此时为字典格式
        mfcc = mfcc_dict['mfcc']#读取其中的变量获得数据，数据依然为numpy格式
        return mfcc

    def get_som(self):#基于MFCC特征数据集获得聚类数据集
        mfcc = self.get_mfcc()
        som=MiniSom(self.x, self.y, input_len=20, sigma=self.sigma, learning_rate=self.learning_rate)#输入拓扑网络结构，输入向量长度对应特征数量，SOM中不同相邻节点的半径定义为0.1，迭代期间权重的的调整幅度定义为0.2
        som.random_weights_init(mfcc)#将SOM的权重初始化为小的标准化随机值
        som.train_random(mfcc, self.epoch)
        som_data = som.quantization(mfcc)#将mfcc特征数据用类簇数据替换
        return som_data#输出聚类数据集

    def get_label(self):
        label_dict = scipy.io.loadmat(self.path2)
        label = label_dict['label']
        return label

class Base_Tools(object):#处理数据的基础工具
    def __init__(self, data, s_data):#输入MFCC特征数据集和聚类后的数据集
        self.data = data
        self.s_data = s_data

    def Z_Score(self):#输入二维数组，标准化处理数据
        z_data = sklearn.preprocessing.scale(self.data, axis=1)
        return z_data

    def get_sort(self):#输入二维数组，升序排列每行数据
        s_data = numpy.sort(self.data)
        return s_data

    def data_unique(self):#输入二维数组，去除重复行
        unique_data = numpy.unique(self.data)
        return unique_data

    def one_hot(self):#输入SOM拓扑网络(x,y)，设计与之对应的独热编码，该独热编码的设定仅仅建立在SOM神经网络的拓扑节点为一维结构
        x = self.data.shape(0)
        y = self.data.shape(1)
        z = x*y
        one_hot_label = numpy.zeros(z,z)
        for i in range(z):
            one_hot_label[i][i] = 1
            
        return one_hot_label

class SOM_Tools(Base_Tools):#处理聚类数据的工具
    def get_clusters(self, label_data):#输入原数据，聚类数据和端点标签
        unique_data = Base_Tools.data_unique(self.s_data)#删除重复行后，二维数组仅由类簇构成
        s_list = []#构建类簇数据列表
        l_list = []#构建类簇数据对应的标签列表
        sequence_list = []#构建序号列表
        for i in unique_data:
            temp = []#构建临时列表
            temp_label = []
            temp_sequence=[]
            for a,b in enumerate(self.s_data):
                if b == i:
                    temp.append(self.s_data[a])
                    temp_label.append([label_data[a]])
                    temp_sequence.append(a)

            sequence_list.append(temp_sequence)
            s_list.append(temp)
            l_list.append(temp_label)

        return s_list, l_list, sequence_list#输出类簇构成的列表和对应的端点列表

    def get_clusters_probability(self, label_data):#输入原数据，聚类数据和端点标签
        clusters_list, clusters_label = self.get_clusters(label_data)#获得类簇列表和对用的端点列表
        probability=[]
        for i in clusters_label:#端点列表由各类簇端点列表构成
            pr=sum(i)/len(i)#由于类簇端点列表由0和1组成，因此可根据长度和和判断语音标签的占比
            probability.append(pr)

        return probability#输出基于各类簇端点列表获得各类簇语音端点占比的概率列表
        
    def get_one_hot(self, label_data):#输入原数据，聚类数据和端点标签
        clusters_list, clusters_label, clusters_sequence = self.get_clusters(label_data)#获得类簇列表和对用的端点列表，以及存有类簇中对应的时间序列序号的列表
        one_hot_data = Base_Tools.one_hot(self.data)#构建独热编码不需要考虑其概率分布大小，独热编码的顺序和类簇顺序没关联
        count = 0#计数器
        one_hot_list = numpy.zeros((len(label_data),len(one_hot_data)))
        for i in clusters_sequence:
            for j in i:
                one_hot_list[j] = one_hot_data[count]

            count += 1

        return one_hot_list#输出基于每个时间节点与类簇的对应关系实现的独热编码