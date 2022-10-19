import torch

#生成从0-11的一维数组
x = torch.arange(12)
#奖数组重新shape为三行四列的排列方式
x = x.reshape(3,4)
#数组里面元素数量
x.numel()
#2个三行四列的三维数组
a = torch.zeros((2,3,4))

#数组转为tensor
x=torch.tensor([1.0,2,4,8])
y=torch.tensor([2,2,2,2])

print(x + y, x - y, x * y, x / y, x ** y)

X=torch.arange(12,dtype=torch.float32).reshape((3,4))
Y=torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
#dim=0表示上下叠加（沿行，行数增加，列数不变）；dim=1表示左右叠加（按列，列数增加，行数不变）
torch.cat((X,Y),dim=0),torch.cat((X,Y),dim=1)