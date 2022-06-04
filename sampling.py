import librosa 
import soundfile
import torch 
import build
import matplotlib.pyplot as plt 
import numpy as np 

def save_audio(filename, mel):
    mel = librosa.db_to_amplitude(mel)
    S = librosa.feature.inverse.mel_to_audio(mel)
    print(S.shape, S.mean(), S.max(), S.min())
    soundfile.write(filename, S, 22050)

def sample_vae(filename, model, zc, dur=1025):
    model.eval()
    print(model)
    z = torch.randn((1, zc, dur))
    y = model.decode(z, None).detach()[0, :, :]
    return y

model = build.saved_model('MUS-173', 'model_99.92307692307692')
y = sample_vae('sample.wav', model, 16)
save_audio('sample.wav', y.numpy())