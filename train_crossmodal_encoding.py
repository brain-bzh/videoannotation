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
from soundnet_model import WaveformCNN8, WaveformCNN5

import argparse

parser = argparse.ArgumentParser(description='Neuromod Movie10 Distillation-transferred encoding model')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=5000, type=int, help='Maximum number of epochs')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha : penalty for Audioset Probas (KLdiv)')
parser.add_argument('--beta', default=1.0, type=float, help='beta : penalty for ImageNet Probas (KLdiv)')
parser.add_argument('--gamma', default=1.0, type=float, help='gamma : penalty for Places Probas (KLdiv)')
parser.add_argument('--delta', default=1.0, type=float, help='delta : penalty for fmri encoding model (MSE)')
parser.add_argument('--epsilon', default=1.0, type=float, help='epsilon : penalty for attention output selection')
parser.add_argument('--ninputfilters', default=16, type=int, help='number of features map in conv1')
parser.add_argument('--expansion', default=1, type=int, help='conv2 will be 32*expansion features, conv3 64*expansion, conv4 128*expansion')
parser.add_argument('--hidden', default=1000, type=int, help='Number of neurons for hidden layer in the encoding model (previous layer has 128*expansion fm)')
parser.add_argument('--nroi_attention', default=None, type=int, help='number of regions to learn using outputattention')
parser.add_argument('--resume', default=None, type=str, help='Path to model checkpoint to resume training')

args = parser.parse_args()

from train_utils import train_kl,test_kl,AudioToEmbeddings,trainloader,valloader,testloader
from pytorchtools import EarlyStopping
from datetime import datetime

from nilearn.plotting import plot_stat_map

from nilearn.regions import signals_to_img_labels

from eval_utils import test_kl_r2

mistroifile = '/home/brain/MIST_ROI.nii.gz'
                     
### hyperparameters 
# 
# alpha : penalty for Audioset Probas (KLdiv)
# beta : penalty for ImageNet Probas (KLdiv)
# gamma : penalty for Places Probas (KLdiv)
# delta : penalty for fmri encoding model (MSE)     
# epsilon : penalty for attention output selection
# ninputfilters : number of features map in conv1
# nroi_attention : number of regions to learn using outputattention
# nfeat : conv2 will be 32*nfeat features, conv3 64*nfeat, conv4 128*nfeat

alpha = args.alpha
beta = args.beta
gamma = args.gamma
delta = args.delta
epsilon = args.epsilon
ninputfilters = args.ninputfilters
nfeat = args.expansion
nroi_attention = args.nroi_attention
fmrihidden = args.hidden

print(args)
destdir = 'cp_{}_{}_{}_{}_{}'.format(alpha,beta,gamma,delta,epsilon)
### Model Setup
if args.resume is not None:
    print("Reloading model {}".format(args.resume))
    old_dict = torch.load(args.resume)
    net = old_dict['model']
    net.load_state_dict(old_dict['net'])
    nfeat = old_dict['nfeat']
    ninputfilters = old_dict['ninputfilters']
else:
    print("Training from scratch")
    net = WaveformCNN5(nfeat=nfeat,ninputfilters=ninputfilters,do_encoding_fmri=True,fmrihidden=fmrihidden,nroi_attention=nroi_attention)

net = net.cuda()
kl_im = nn.KLDivLoss(reduction='batchmean')
kl_audio = nn.KLDivLoss(reduction='batchmean')
kl_places = nn.KLDivLoss(reduction='batchmean')
mseloss = nn.MSELoss(reduction='mean')

### Optimizer and Schedulers
optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9)
lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2,patience=10,threshold=1e-4,cooldown=2)

early_stopping = EarlyStopping(patience=15, verbose=True)
nbepoch = args.epochs

#### Simple test just to check the shapes

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
        fmri = torch.Tensor(onesample['fmri']).view(bsize,-1).cuda()

        print(wav.shape)

        print(imnet.shape,places.shape,audioset.shape,fmri.shape)
        obj_p,scene_p,audio_p,fmri_p = net(wav)
        print(obj_p.shape,scene_p.shape,audio_p.shape,fmri_p.shape)

### Main Training Loop 
startdate = datetime.now()

train_loss = []
val_loss = []
for epoch in tqdm(range(nbepoch)):
    train_loss.append(train_kl(epoch,trainloader,net,optimizer,kl_im,kl_audio,kl_places,mseloss=mseloss,alpha=alpha,beta=beta,gamma=gamma,delta=delta,epsilon=epsilon))
    val_loss.append(test_kl(epoch,valloader,net,optimizer,kl_im,kl_audio,kl_places,mseloss=mseloss,alpha=alpha,beta=beta,gamma=gamma,delta=delta,epsilon=epsilon))
    #print("Train : {}, Val : {} ".format(train_loss[-1],val_loss[-1]))
    lr_sched.step(val_loss[-1])

    # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
    early_stopping(val_loss[-1], net)
    
    print(np.argmax(net.maskattention.detach().cpu().numpy(),axis=0))
    if early_stopping.early_stop:
        print("Early stopping")
        break

test_loss = test_kl(1,testloader,net,optimizer,kl_im,kl_audio,kl_places,mseloss=mseloss,alpha=alpha,beta=beta,gamma=gamma,delta=delta,epsilon=epsilon)
#print("Test Loss : {}".format(test_loss))

enddate = datetime.now()

if not os.path.isdir(destdir):
    os.mkdir(destdir)

## Reload best model 
net.load_state_dict(torch.load('checkpoint.pt'))



dt_string = enddate.strftime("%Y-%m-%d-%H-%M-%S")
str_bestmodel = os.path.join(destdir,"{}.pt".format(dt_string))
str_bestmodel_plot = os.path.join(destdir,"{}_{}_{}.png".format(dt_string,ninputfilters,nfeat))
str_bestmodel_nii = os.path.join(destdir,"{}_{}_{}.nii.gz".format(dt_string,ninputfilters,nfeat))



# Remove temp file 
os.remove('checkpoint.pt')

r2model = test_kl_r2(testloader,net,kl_im,kl_audio,kl_places,mseloss=mseloss)
r2model[r2model<0] = 0
print("mean R2 score on test set  : {}".format(r2model.mean()))

print("max R2 score on test set  : {}".format(r2model.max()))

print("Training time : {}".format(enddate - startdate))

## Prepare data structure for checkpoint
state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'train_loss' : train_loss,
            'val_loss' : val_loss,
            'test_loss' : test_loss,
            'r2' : r2model,
            'r2max' : r2model.max(),
            'r2mean' : r2model.mean(),
            'nfeat' : nfeat,
            'training_time' : enddate - startdate,
            'ninputfilters' : ninputfilters,
            'model' : net
        }


### Plot the loss figure
f = plt.figure(figsize=(20,10))

ax = plt.subplot(2,1,2)

plt.plot(state['train_loss'])
plt.plot(state['val_loss'])
plt.legend(['Train','Val'])
plt.title("Mean $R^2=${}, Max $R^2=${}".format(r2model.mean(),r2model.max()))

### R2 figure 
r2_img = signals_to_img_labels(r2model.reshape(1,-1),mistroifile)

ax = plt.subplot(2,1,1)

plot_stat_map(r2_img,display_mode='z',cut_coords=8,figure=f,axes=ax)
f.savefig(str_bestmodel_plot)
r2_img.to_filename(str_bestmodel_nii)
plt.close()





torch.save(state, str_bestmodel)
