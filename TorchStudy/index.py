import torch
a = torch.rand(4,3,28,28)
print(a[0].shape)
b = a[0,0].shape
c = a[:2,1:,-1:,:].shape     #dim1:0,1;dim2:1,2;dim3:27;dim4:0-27
d = a[:,:,0:28:2,0:28:2].shape  #隔行取样
e = a.index_select(0,torch.tensor([0,2])).shape  #针对0dim上的0和2索引进行采样
f = a[0,...].shape  #a[0] ...表示尽可能采样
mask = a.ge(0.5)
torch.masked_select(a,mask)    #选取a中大于0.5的数，并返回打平后的数
