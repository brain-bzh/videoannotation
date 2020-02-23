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
from soundnet_model import WaveformCNN
from train_utils import testset
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

            fmri = onesample['fmri'].cuda()

            # Forward pass
            _,_,_,fmri_p = net(wav)

            all_fmri.append(fmri.cpu().numpy())
            all_fmri_p.append(fmri_p.cpu().numpy())
            


    r2_model = r2_score(np.vstack(all_fmri),np.vstack(all_fmri_p),multioutput='raw_values')
    return r2_model