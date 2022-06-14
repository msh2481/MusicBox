import librosa 
import soundfile
import torch 
import build
import matplotlib.pyplot as plt 
import numpy as np 
from dct import isdct

def save_audio(filename, spec):
    # spec = librosa.db_to_amplitude(spec)
    # S = librosa.feature.inverse.mel_to_audio(mel)
    S = isdct(spec, frame_step=256)
    soundfile.write(filename, S, 22050)

model = build.saved_model('MUS-173', 'model_99.92307692307692')
y = model.generate().detach().cpu()
save_audio('sample.wav', y.numpy())