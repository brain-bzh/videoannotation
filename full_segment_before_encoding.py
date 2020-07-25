import os, librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import model_utils as model

from soundnet_model import SoundNet8_pytorch

path = '/home/maelle/Database/cneuromod/movie10/stimuli/life'
SoundNet8_params = './sound8.pth'
sample_rate = 22050
gpool = nn.AdaptiveAvgPool2d((1,1))

soundnet = model.load_net_pretrained(SoundNet8_pytorch, SoundNet8_params, cuda=True, num_cuda = 0)

for segment in os.listdir(path):
    name, ext = os.path.splitext(segment)
    if ext == '.wav':
        print(segment)
        wavfile = os.path.join(path, segment)
        
        audio_lentgh = librosa.core.get_duration(filename = wavfile)
        print(audio_lentgh, audio_lentgh/1.49)

        (waveform, _) = librosa.core.load(wavfile, sr=sample_rate, mono = True, duration=audio_lentgh)
        #print(f'1 : ', waveform.shape)
        x = np.reshape(waveform, (1,1,-1,1))
        print(f'2 : ',x.shape)
        x = torch.from_numpy(x).cuda(0)
        y = soundnet(x).cpu()
        print(f'3 : ',audio_lentgh, ' -> ',y.shape)
        #y = gpool(y)
        #print(f'4 : ',y.shape)
        #y = y.view(1, -1).detach().numpy()
        #print(f'5 : ',y.shape)

