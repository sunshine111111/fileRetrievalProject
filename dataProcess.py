import os
import pandas as pd
import torch

#构建数据集
os.makedirs(os.path.join('C:/Users/zhongxing/Desktop/lan/project','data'),exist_ok=True)
data_file = os.path.join('C:/Users/zhongxing/Desktop/lan/project','data','house_tiny.csv')
with open(data_file,'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每⾏表⽰⼀个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

#处理缺失值:"NaN"项代表缺失值，处理缺失数据，一般包括插值法和删除法
#data.iloc:左闭右开，取所有行，列数取指定的列数
inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
#fillna()方法，能够使用指定的方法填充NA/NaN值。
#inputs.mean()是取平均值
#总的意思是取平均值去填充NA/NaN值
inputs = inputs.fillna(inputs.mean())
print(inputs)

#将类别值转为离散值（0,1），dummy_na=True，表示将“NaN"单独视为一个类别。dummy_na=False，表示不考虑“NaN"类别，直接置为0
inputs = pd.get_dummies(inputs,dummy_na=True)
print(inputs)

#将数值型的inputs和outputs，转换为张量格式
X,y = torch.tensor(inputs.values),torch.tensor(outputs.values)
print(X,y)