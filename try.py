from models import ConvBlock
from train import generate

m = ConvBlock(1, 1, 1, shift=1)
y = generate(m, 10)
