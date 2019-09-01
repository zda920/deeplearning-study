#矩阵相乘 torch.mm 只用于2d torch.matmul() 推荐
#数字相乘 *
#[4,1,32,64] 与 [4,3,64,16]  得[4,3,32,16]
grad.clamp(0,10)  #限制在0-10
grad.clamp(10)   #限制小于10的
