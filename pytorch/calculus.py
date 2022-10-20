import numpy as np
import torch
import matplotlib.pyplot as plt

def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f,x,h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    #f：print输出格式化；.5f:5位浮点数
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

#####标量梯度计算
#requires_grad=True 的作用是让 backward 可以追踪这个参数并且计算它的梯度。
#如果我们需要计算某个Tensor的导数，那么我们需要设置其参数.requires_grad属性为True。
x = torch.arange(4.0,requires_grad=True)
#grad: 该Tensor的梯度值， 每次在计算backward时都需要将前一时刻的梯度归零，否则梯度值会一直累加。
print(x,x.grad)

y = 2 * torch.dot(x,x)
#反向传播函数:如果你的Y值是个标量，那么直接使用backward()；
#如果y是个向量或者矩阵，就得在backward()里面计入参数
#当完成计算后通过调用 .backward()，自动计算所有的梯度， 这个张量的所有梯度将会自动积累到 .grad 属性。
y.backward()
#这里y=2*(x*x),y的导数就是y=4x；
# x.grad的值就是，当x=tensor([0,1,2,3]),将x的这四个值分别带到导数函数里面，计算出x的梯度值就是tensor([ 0.,  4.,  8., 12.])
print(x.grad)

####计算另一个函数的梯度值
#默认情况下，torch会累积梯度，清除上一个函数计算的梯度值
x.grad.zero_()
y = x.sum
y.backward()
#y = x.sum，这个函数的导数是1，所以x.grad的值为tensor([1., 1., 1., 1.])
print(x.grad)










