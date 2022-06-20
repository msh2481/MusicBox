import librosa 
import soundfile
import torch 
import build
import matplotlib.pyplot as plt 
import numpy as np 
from dct import sdct, isdct


model = build.saved_model("MUS-262", "model_70.44444444444444", init_with=(4, 1, 256, 256), strict=True)
print(model)

sample = next(iter(loader))[0].cuda()
model.eval()
t, aux = model.encode(sample)
model.eval()
back = model.decode(t, aux)
print(criterion(input=back, target=sample, aux=(t,aux), info=None))