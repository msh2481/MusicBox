import librosa 
import soundfile
import torch 
import build
import matplotlib.pyplot as plt 
import numpy as np 

def save_audio(filename, mel):
    mel = librosa.db_to_amplitude(mel)
    S = librosa.feature.inverse.mel_to_audio(mel)
    soundfile.write(filename, S, 22050)

model = build.saved_model('MUS-173', 'model_99.92307692307692')
y = model.generate().detach().cpu()
save_audio('sample.wav', y.numpy())