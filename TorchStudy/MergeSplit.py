import torch
a1 = torch.rand(4,3,32,32)
a2 = torch.rand(4,3,32,32)
torch.cat([a1,a2],dim=1)
torch.stack([a1,a2],dim=0)   #创建新的维度成为  [2,4,3,32,32] 维度必须一致
aa, bb = a1.split([2,1],dim=0)  #在维度0上按长度2，1拆分成两个
cc, dd = a2.chunk(2,dim=0)      #按数量进行拆分
