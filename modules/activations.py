import math
import torch
import torch.nn as nn


class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, inputs):
        return gelu_(inputs)


# 高斯误差线性单元
def gelu(input_tensors):
    # torch.erf Gauss error function
    cdf = 0.5 * (1.0 + torch.erf(input_tensors / math.sqrt(2.0)))
    return input_tensors * cdf


# 论文给出的gelu激活函数的近似计算
def gelu_(input_tensors):
    cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2. / math.pi) * (input_tensors + 0.044715 * torch.pow(input_tensors, 3))))
    return input_tensors * cdf


if __name__ == '__main__':
    x = torch.rand(3, 4)
    print(x)
    print(gelu_(x))
    print(gelu(x))