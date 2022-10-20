import random
import torch
import matplotlib.pyplot as plt

#######生成数据集
###生成y=Xw+b+噪声
def synthetic_data(w,b,num_examples):
    #从一个标准正态分布N～(0,1)，提取一个(num_examples,len(w))的矩阵
    X = torch.normal(0,1,(num_examples,len(w)))
    #torch.matmul是tensor的乘法，输入可以是高维的
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.01,y.shape)
    #y.reshape((-1,1))，将y变成1列的格式，行数自动计算
    return X,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)
print('features:', features[0],'\nlabel:', labels[0])

##⽣成⼤⼩为batch_size的⼩批量。每个⼩批量包含⼀组特征和标签。
def data_iter(batch_size,features,labels):
    num_examples = len(features)
    print(num_examples)
    #range(num_examples)：生成一个(0到num_examples-1)的可迭代对象
    #list()将可迭代对象转为list列表
    indices = list(range(num_examples))
    #这些样本是随机读取的，没有特定的顺序
    #shuffle():让训练数据集中的数据打乱顺序，然后一个挨着一个地(for i in indices)生成训练数据对
    random.shuffle(indices)
    #随机取一个大小为batch_size的一组特征和标签
    # 范围从0到num_examples，步长是batch_size
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

batch_size = 10
for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break

#####初始化模型参数
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

#####定义模型
def linreg(X,w,b):
    #广播机制：当我们用一个向量加一个标量时，标量会被加到向量的每一个分量上。
    return torch.matmul(X,w) + b

#####定义损失函数
def squared_loss(y_hat,y):
    #这里需要将真实值y的形状转换为和预测值y_hat的形状相同
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

#####定义优化算法,更新权重向量和偏置项
def sgd(params,lr,batch_size):
    #小批量随机梯度下降
    #torch.no_grad()，所有计算得出的tensor的requires_grad都自动设置为False。
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

#####训练
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        #X和y的小批量损失
        l = loss(net(X,w,b),y)
        #损失函数求梯度向量
        l.sum().backward()
        #使用参数的梯度更新参数
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print([w, b])
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')