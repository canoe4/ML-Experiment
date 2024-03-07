# 验证线性回归


# 导包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#导入数据并可视化
data_set = pd.read_csv('Salary_Data.csv')
data_set = data_set.take(np.random.permutation(30))#随机排列
data_set

data_set.plot.scatter(x='YearsExperience',y='Salary')
plt.show()

data_set = np.array(data_set)
data_set

# 分割数据集为训练集和测试集
data_train,data_test = np.split(data_set,[20],)#水平切割
n_train = np.shape(data_train)[0] #训练集的个数
n_test = np.shape(data_test)[0]

x_train = data_train[:,:1]
y_train = data_train[:,1:2]

x_test = data_test[:,:1]
y_test = data_test[:,1:2]


# 列出模型 y = wx + b
#对参数进行初始化
w = np.ones(n_train).reshape(n_train,1) #20列1行的矩阵
b = 1.0

lr = 0.00001 #学习率

num_iter = 100000 #迭代次数


# 使用梯度下降求解参数
for i in range(num_iter):
    y_hat = w*x_train + b       #y_hat是预测值
    sum_w = np.sum((y_train-y_hat)*(-x_train))    
    sum_b = np.sum((y_train-y_hat)*(-1))
    det_w = 2 *sum_w       
    det_b = 2 *sum_b
    w = w - lr * det_w
    b = b - lr * det_b
print(w)
print(b)



# 可视化
fig,ax = plt.subplots()
ax.scatter(x_train,y_train)
ax.scatter(x_test,y_test,c = 'r')
ax.plot([i for i in range(0,12)],[w[0,0]*i+b for i in range(0,12)])
ax.set_xlabel('YearsExperience')
ax.set_ylabel('Salary')
plt.title('y = w*x + b')
plt.legend(('Training set','Test set'),loc = 'upper left')
plt.show()
