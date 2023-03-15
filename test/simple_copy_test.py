import torch
import time
import torch_xla
import torch_xla.core.xla_model as xm

xla_device = xm.xla_device()
cuda_device = 'cuda'


a = torch.rand([1,1024,1024], device=cuda_device)
# b = torch.ones([3,2,1], device=cuda_device)
# xm.mark_step()
a = a.to(xla_device)
# c =  b + a

xm.mark_step()
print(a)
# del a
# del b

# xm.mark_step()
torch.cuda.synchronize()
# torch.cuda.empty_cache()
# print(torch.cuda.memory_summary())
# print(b)
# del a

# c = b.to(xla_device)

# d = c.to(cuda_device)
# print(a)
# print(b)
# print(c)
print(torch.cuda.memory_summary())

# torch.cuda.synchronize()
# torch.cuda.empty_cache()
# print(torch.cuda.memory_summary())
# print(b)
# a += 1
# print(a) 
# print(b) 
# del a

# print(b)