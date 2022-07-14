from dataclasses import replace
import torch

logits = torch.randn((10, 256))
tokens = torch.multinomial(torch.softmax(logits, dim=1), 3)
print(tokens)
logits = logits.gather(1, tokens)
print(logits.shape)
print(logits)