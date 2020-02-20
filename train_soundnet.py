## Author : Nicolas Farrugia, February 2020
from matplotlib import pyplot as plt 
import torch
from torchvision.io import read_video,read_video_timestamps
import torchvision.transforms as transforms
from utils import convert_Audio
import torch.nn as nn
from importlib import reload
from tqdm import tqdm
import os 
import sys
import numpy as np 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import librosa
import soundfile
from soundnet_model import SoundNet8_pytorch,SmallerWaveCNN,WaveformCNN
from train_utils import train_kl,test_kl,AudioToEmbeddings,trainloader,valloader,testloader




net = WaveformCNN(nfeat=4,ninputfilters=16)
net = net.cuda()
optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1,patience=5,threshold=1e-4)
#optimizer = torch.optim.Adam(net.parameters())

kl_im = nn.KLDivLoss(reduction='batchmean')
kl_audio = nn.KLDivLoss(reduction='batchmean')
kl_places = nn.KLDivLoss(reduction='batchmean')


#### Simple Test case just to check the shapes

if False:
    from train_utils import dataset

    with torch.no_grad():
        
        onesample = dataset.__getitem__(25)

        bsize = 1

        # load data
        wav = torch.Tensor(onesample['waveform']).view(bsize,1,-1,1).cuda()
        places = torch.Tensor(onesample['places']).view(bsize,-1,1,1).cuda()
        audioset = torch.Tensor(onesample['audioset']).view(bsize,-1,1,1).cuda()
        imnet = torch.Tensor(onesample['imagenet']).view(bsize,-1,1,1).cuda()

        print(wav.shape)
        obj_p,scene_p,audio_p = net(wav)
        print(obj_p.shape,scene_p.shape,audio_p.shape)
        print(imnet.shape,places.shape,audioset.shape)


nbepoch = 50
train_loss = []
val_loss = []
for epoch in tqdm(range(nbepoch)):
    train_loss.append(train_kl(epoch,trainloader,net,optimizer,kl_im,kl_audio,kl_places))
    val_loss.append(test_kl(epoch,valloader,net,optimizer,kl_im,kl_audio,kl_places))
    print("Train : {}, Val : {} ".format(train_loss[-1],val_loss[-1]))
    lr_sched.step(val_loss[-1])

print("Test Loss : {}".format(test_kl(1,testloader,net,optimizer,kl_im,kl_audio,kl_places)))

plt.plot(range(nbepoch),train_loss)
plt.plot(range(nbepoch),val_loss)
plt.show()





