import torch
a = torch.rand(4,1,28,28)
#view = reshape
a.view(4,28*28)  #关注img的像素
a.view(4*28,28)  #关注img的行像素
a.view(4*1,28,28) #关注img的feature map
b = a.unsqueeze(0).shape  #维度插入  在0处插入一维，其余不变   正的索引在之前，负的索引在之后
c = a.squeeze()    #除去所有shape上数字为1的维度
d = a.squeeze(0)      #除去索引为0 的维度，只有shape上数字为1才会减去[1,32,1,1]
e = a.expand(4,32,4,4).shape   #只有shape上数字为1的维度才支持扩张，不为1则不变，如果想不变用-1
f = a.repeat(4,32,1,1).shape   #数字表示拷贝的次数
a1 = a.transpose(1,3)     #交换1，3的维度
a2 = a.transpose(1,3).contiguous()  #将数据连续
g =  a.permute(0,2,3,1).shape    #按照给出的数字上进行交换，（）里的数字表示各个维度
# boradcast 相当于unsqueeze + expand ，从低维开始
