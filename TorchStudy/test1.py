import torch
a = torch.rand(2,3,4)
print(a)
img = torch.rand(2,3,28,28)  #num , channal ,height, weight
print(img.numel())