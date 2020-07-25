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
from pytorchtools import EarlyStopping
from datetime import datetime

from sklearn.metrics import r2_score

from nilearn.plotting import plot_stat_map

from nilearn.regions import signals_to_img_labels
import sys

def test_kl_r2(testloader,net,kl_im,kl_audio,kl_places,mseloss):
    all_fmri = []
    all_fmri_p = []
    net.eval()
    with torch.no_grad():
        for onesample in testloader:

            bsize = onesample['waveform'].shape[0]
            
            # load data
            wav = torch.Tensor(onesample['waveform']).cuda()

            wav = wav.view(bsize,1,-1,1)

            fmri = onesample['fmri'].view(bsize,1,-1).cuda()

            # Forward pass
            _,_,_,fmri_p = net(wav)
            
            all_fmri.append(fmri.cpu().numpy().reshape(bsize,-1))
            all_fmri_p.append(fmri_p.cpu().numpy().reshape(bsize,-1))
            


    r2_model = r2_score(np.vstack(all_fmri),np.vstack(all_fmri_p),multioutput='raw_values')
    return r2_model


def test_r2(testloader,net,mseloss):
    all_fmri = []
    all_fmri_p = []
    net.eval()
    with torch.no_grad():
        for (wav,audioset,imagenet,places,fmri) in testloader:

            bsize = wav.shape[0]
            
            # load data
            wav = torch.Tensor(wav).view(1,1,-1,1).cuda()

            fmri = fmri.view(bsize,-1).cuda()

            # Forward pass
            fmri_p = net(wav).permute(2,1,0,3).squeeze()

            #Cropping the end of the predicted fmri to match the measured bold
            fmri_p = fmri_p[:bsize]
            
            all_fmri.append(fmri.cpu().numpy().reshape(bsize,-1))
            all_fmri_p.append(fmri_p.cpu().numpy().reshape(bsize,-1))
            


    r2_model = r2_score(np.vstack(all_fmri),np.vstack(all_fmri_p),multioutput='raw_values')
    return r2_model