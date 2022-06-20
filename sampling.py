import librosa 
import soundfile
import torch 
import build
import matplotlib.pyplot as plt 
import numpy as np 
from dct import sdct, isdct

def save_audio(filename, spec):
    # spec = librosa.db_to_amplitude(spec)
    # S = librosa.feature.inverse.mel_to_audio(mel)
    S = isdct(spec, frame_step=256)
    soundfile.write(filename, S, 22050)

spec = build.dataset('dataset_v4')[0][0]


model = build.saved_model('MUS-247', 'model_1.0')

# model.eval()

# code, aux = model.encode(spec.unsqueeze(0))
# print(code)
# print(aux)

# spec = model.decode(code, None)

# spec = spec[0][0].detach().numpy()
# print(spec.shape)
# plt.imshow(librosa.power_to_db(spec ** 2))
# plt.show()
# save_audio('check.wav', spec)

# # y = model.generate().detach().cpu()
# # save_audio('sample.wav', y.numpy())