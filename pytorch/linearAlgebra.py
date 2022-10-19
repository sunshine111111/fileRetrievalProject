import torch

###标量
#仅包含一个数值的叫标量，变量表示未知的标量值
#只有一个元素的张量，就是一个标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y,x * y,x / y,x ** y)

###向量：一维张量,一维数组
x = torch.arange(4)
#内置len()函数：张量的长度
len(x)

###矩阵：矩阵将向量从一阶推广到二阶
A = torch.arange(20).reshape(5,4)
print(A)
#矩阵的转置
A.T
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B == B.T
print(B == B.T)

###张量
#向量是标量的推广，矩阵是向量的推广一样，张量可以构建具有更多轴的数据结构。
#向量是一阶张量，矩阵是二阶张量。
#张量用特殊字体的大写字母表示（例如，X,Y和Z),索引机制与矩阵类似
X = torch.arange(24).reshape(2,3,4)

A = torch.arange(20,dtype=torch.float32).reshape(5,4)
B = A.clone()
#这里的矩阵相加是矩阵中相同位置的元素相加
print(A,A+B)
#这里的矩阵相乘是矩阵中相同位置的元素相乘
print(A*B)

#张量乘以或者加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或者相乘
a = 2
X = torch.arange(24).reshape(2,3,4)
a + X, (a * X).shape

###降维
#求和
x = torch.arange(4,dtype=torch.float32)
print(x,x.sum())

#A是5行4列的数组，A.sum()是将矩阵里面所有的值相加
A.shape,A.sum()

#列数不变，将固定列的每行数据相加
A_sum_axis0 = A.sum(axis=0)
print(A,A_sum_axis0,A_sum_axis0.shape)

#行数不变，将固定行的每列数据相加
A_sum_axis1 = A.sum(axis=1)
print(A,A_sum_axis1,A_sum_axis1.shape)

#对矩阵的所有元素进行求和
A.sum(axis=[0,1])

#求平均值
A.mean(),A.sum()/ A.numel()

#沿指定轴降低张量的维度
A.mean(axis=0),A.sum(axis=0)/ A.shape[0]

sum_A = A.sum(axis=1, keepdims=True)
print(sum_A)

A / sum_A

#不会改变维度，每行数据相加，每行新的数据是同一列里面，当前行的数值加上前面行的数值
A.cumsum(axis=0)

y = torch.ones(4,dtype = torch.float32)
#点积
print(x,y,torch.dot(x,y))

#向量积
print(A,x,torch.mv(A,x))

#矩阵乘法
B = torch.ones(4,3)
print(A,B,torch.mm(A,B))

#范数:距离的度量
u = torch.tensor([3.0,-4.0])
#L2范数：元素平方和，再取平方根
torch.norm(u)
#L1范数：元素绝对值之和
torch.abs(u),sum()
#矩阵元素平方和的平方根
torch.norm(torch.ones((4,9)))










