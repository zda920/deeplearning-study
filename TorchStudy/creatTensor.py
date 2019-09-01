import torch
torch.tensor([2.,3.2])  #接受现成的data
torch.Tensor(2,3,4)   # 接受shape,dim1,dim2,dim3  默认为floatTensor
print(torch.randint(1,10,[3,3]))
torch.randn(3,3)  #正态分布 N(0,1)
torch.normal(mean=torch.full([10],0), std=torch.arange(1,0,-0.1))  # N(u,std)
torch.linspace(0,10,steps=4)       #均匀切割0-10
torch.logspace(0,1,steps=10)       #以10，或e为基数，切割10份
torch.randperm(10)   #随机打散