from Tools import Main_Tools, SOM_Tools

path1 = 'm.mat'#mfcc文件路径
path2 = 'l.mat'#端点标签路径
x = 1#som拓扑结构
y = 1
sigma=0.05
learning_rate=0.1
epoch = 100#迭代次数

if __name__ == "__main__":
    get_data = Main_Tools(path1, path2, x, y, sigma, learning_rate, epoch)
    mfcc_data = get_data.get_mfcc()#获取mfcc数据
    som_data = get_data.get_som()#获取聚类数据
    label_data = get_data.get_label()#获取标签数据

    clusters = SOM_Tools(mfcc_data, som_data)
    one_hot_label = clusters.get_one_hot(label_data)