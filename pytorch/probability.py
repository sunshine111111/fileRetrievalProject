import torch
from torch.distributions import multinomial

fair_probs = torch.ones([6]) / 6
#Multinomial函数第一个参数是试验次数，第二个参数是事件概率，非负的，有限的，并且归一化处理过的，所有概率和为1
counts = multinomial.Multinomial(5,fair_probs).sample()
#结果为tensor([3., 0., 0., 0., 1., 1.])，表示投掷5次骰子，出现1的次数为3，出现5的次数为1，出现6的次数为1，加起来是5次
print(counts)

counts = multinomial.Multinomial(1000,fair_probs).sample()
##相对频率作为估计值
print(counts,counts / 1000)

#以概率fair_probs进行10次实验，生成一个多项式分布，然后再此基础上，采样500次（相当于进行了500组实验，每组需要扔10次骰子）
counts = multinomial.Multinomial(10,fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
print(counts,cum_counts)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)