import torch

a = torch.range(1, 36).reshape(4, 3, 3)
print(a)
b = torch.ones_like(a)
_mean = torch.mean(a, dim=[0, 2])
_mean = _mean.unsqueeze(0)
_mean = _mean.unsqueeze(2)
b = b * _mean
print(b)
# a = torch.mean(a, dim=[0, 1, 2])
# print(a)
# b[:, :, :] = a
# a = b
# print(a)
# print(b)
