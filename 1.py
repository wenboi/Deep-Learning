# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:20:20 2018

@author: 文博
"""

from numpy import *
import os

path = r'..\data'
training_sample = 'Logistic_Regression-trainingSample.txt'
testing_sample = 'Logistic_Regression-testingSample.txt'

# 从文件中读入训练样本的数据
def loadDataSet(p, file_n):
    dataMat = []
    labelMat = []
    fr = open(os.path.join(p, file_n))
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 三个特征x0, x1, x2, x0=1
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))  # 样本标签y
    return dataMat, labelMat

def sigmoid(X):
    return 1.0/(1+exp(-X))

# 梯度下降法求回归系数a
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             # 转换成numpy中的矩阵, X, 90 x 3
    labelMat = mat(classLabels).transpose()  # 转换成numpy中的矩阵, y, 90 x 1
    m, n = shape(dataMatrix)  # m=90, n=3
    alpha = 0.001  # 学习率
    maxCycles = 1000
    weights = ones((n, 1))  # 初始参数, 3 x 1
    for k in range(maxCycles):              # heavy on matrix operations
        h = sigmoid(dataMatrix * weights)     # 模型预测值, 90 x 1
        error = h - labelMat              # 真实值与预测值之间的误差, 90 x 1
        temp = dataMatrix.transpose() * error  # 所有参数的偏导数, 3 x 1
        weights = weights - alpha * temp  # 更新权重
    return weights

# 分类效果展示，参数weights就是回归系数
def plotBestFit(weights,training_sample):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet(path, training_sample)
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]  # x2 = f(x1)
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
# 测试函数    
def predict_test_sample():
    A = [5.262118, 0.60847797, -0.75168429]  # 上面计算出来的回归系数a
    dataArr, labelMat = loadDataSet(path, testing_sample)  
    h_test = sigmoid(mat(dataArr) * mat(A).transpose())  # 将读入的数据和A转化成numpy中的矩阵
    print(h_test)  # 预测的结果
    plotBestFit(A,testing_sample)

# 训练函数
def test_logistic_regression():
    dataArr, labelMat = loadDataSet(path, training_sample)  # 读入训练样本中的原始数据
    A = gradAscent(dataArr, labelMat)  # 回归系数a的值
    h = sigmoid(mat(dataArr)*A)  # 预测结果h(a)的值
    print(dataArr, labelMat)
    print(A)
    print(h)
    plotBestFit(A.getA(),training_sample)
    
print("训练数据")

test_logistic_regression()

print("测试数据")

predict_test_sample()