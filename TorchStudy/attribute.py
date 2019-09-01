a.norm(1)
b.norm(2,dim=1)  #2范数 平方和的根号
a.argmax()  #打平数据 求最大值
a.argmax(dim=1)  #在维度1上求最大值，返回其维度一上的位置
a.argmax(dim=1,keepdim=True)  #保持维度一致
a.topk(3,dim=1)   #返回最大的k个的位置   largest=false 则返回最小的k个的位置
a.kthvalue(8,dim=1)  #返回第k小的位置
a.where(condition,x,y)  #if condition x else y
torch.gather(input,dim,index)  #在维度0上，按照index来聚集input矩阵上的数据