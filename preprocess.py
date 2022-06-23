import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch

from dct import isdct, sdct

K = 100
L = 2**16
X = torch.zeros((0, 1, L), dtype=torch.float32)
y = torch.zeros((0,), dtype=torch.long)


def normalize(x):
    return (x - x.mean()) / x.std()


for cl_num, cl in enumerate(os.listdir("Data/dataset")):
    tensors = torch.zeros((100, 1, L))
    dr = f"Data/dataset/{cl}/"
    print("             dr = ", dr, flush=True)
    signal = None
    for i, f in enumerate(os.listdir(dr)):
        try:
            signal = librosa.load(dr + f, mono=True)[0][:L]
            # spec = librosa.feature.melspectrogram(y=signal, n_mels=128, fmax=8192)
            # S = librosa.power_to_db(spec, ref=np.max)
            # frame = 256
            # S = sdct(signal[:L], frame, frame)
            # wave = isdct(S, frame_step=frame)
            # soundfile.write('check.wav', wave, 22050)
            # plt.imshow(librosa.power_to_db(np.abs(S) ** 2))
            # plt.savefig('sdct.png')
            # exit(0)
        except Exception as err:
            print(err)
            assert signal is not None
        tensors[i] = normalize(torch.tensor(signal, dtype=torch.float))
    X = torch.cat((X, tensors), dim=0)
    y = torch.cat((y, torch.full((100,), cl_num)))

print(X.shape, X.dtype, X.mean(), X.std())
print(y.shape, y.dtype, y.min(), y.max())
torch.save(X, "X_v5.p")
torch.save(y, "y_v5.p")
