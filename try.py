from models import *

m = CausalConv(1, 1, 2, 1, True, 1)
x = torch.randn((10, 1, 100))
y = m(x)
print(y.shape)