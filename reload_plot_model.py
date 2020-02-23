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

modelfile = sys.argv[1]

modeldict = torch.load(modelfile)

#ninputfilters = modelfile['ninputfilters']
#nfeat = modelfile['nfeat']

testloader = DataLoader(testset,batch_size=512)

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

### Model Setup
#net = WaveformCNN(nfeat=nfeat,ninputfilters=ninputfilters,do_encoding_fmri=True)
net = modeldict['model']
net = net.cuda()

kl_im = nn.KLDivLoss(reduction='batchmean')
kl_audio = nn.KLDivLoss(reduction='batchmean')
kl_places = nn.KLDivLoss(reduction='batchmean')
mseloss = nn.MSELoss(reduction='mean')

checkpoint_path = 'bestmodel.pt'
## Reload best model 
net.load_state_dict(modeldict['net'])


r2model = test_kl_r2(testloader,net,kl_im,kl_audio,kl_places,mseloss=mseloss)
r2model[r2model<0] = 0
print("mean R2 score on test set  : {}".format(r2model.mean()))

print("max R2 score on test set  : {}".format(r2model.max()))

### Plot the loss figure
f = plt.figure()

ax = plt.subplot(2,1,2)

plt.plot(modeldict['train_loss'])
plt.plot(modeldict['val_loss'])
plt.legend(['Train','Test'])
plt.title("Mean $R^2=${}, Max $R^2=${}".format(r2model.mean(),r2model.max()))

### R2 figure 
r2_img = signals_to_img_labels(r2model.reshape(1,-1),'/home/nfarrugi/git/MIST_parcellation/MIST_parcellation/Parcellations/MIST_ROI.nii.gz')

ax = plt.subplot(2,1,1)

plot_stat_map(r2_img,display_mode='z',cut_coords=8,figure=f,axes=ax)
plt.show()
r2_img.to_filename('r2_img.nii.gz')