from telnetlib import SE
from models import *

# m = Padded((1, 1), Identity())
m = Padded((1, 1), Identity())
s = repr(m)
# print(eval(s))
# print(m)
# print(repr(m))
print(module_description(m))