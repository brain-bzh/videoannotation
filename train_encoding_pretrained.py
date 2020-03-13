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
from soundnet_model import SoundNetEncoding

import argparse

parser = argparse.ArgumentParser(description='Neuromod Movie10 Distillation-transferred encoding model')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=5000, type=int, help='Maximum number of epochs')
parser.add_argument('--hidden', default=1000, type=int, help='Number of neurons for hidden layer in the encoding model (previous layer has 128*expansion fm)')
parser.add_argument('--nroi_attention', default=None, type=int, help='number of regions to learn using outputattention')
parser.add_argument('--resume', default=None, type=str, help='Path to model checkpoint to resume training')

args = parser.parse_args()

from train_utils import AudioToEmbeddings

from train_utils import trainloader,valloader,testloader,dataset

from train_utils import train,test

from pytorchtools import EarlyStopping
from datetime import datetime

from nilearn.plotting import plot_stat_map

from nilearn.regions import signals_to_img_labels

from eval_utils import test_r2

mistroifile = '/home/brain/MIST_ROI.nii.gz'
                     
nroi_attention = args.nroi_attention
fmrihidden = args.hidden

print(args)


destdir = 'cp_{}'.format(fmrihidden)
### Model Setup
if args.resume is not None:
    print("Reloading model {}".format(args.resume))
    old_dict = torch.load(args.resume)
    net = old_dict['model']
    net.load_state_dict(old_dict['net'])
else:
    print("Training from scratch")
    net = SoundNetEncoding(pytorch_param_path='./sound8.pth',fmrihidden=fmrihidden,nroi_attention=nroi_attention)

net = net.cuda()
mseloss = nn.MSELoss(reduction='sum')

### Optimizer and Schedulers
#optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr = 0.1)
lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2,patience=10,threshold=1e-4,cooldown=2)

early_stopping = EarlyStopping(patience=15, verbose=True)

nbepoch = args.epochs

if False:
    
    with torch.no_grad():
        
        onesample = dataset.__getitem__(25)

        bsize = 1

        # load data
        wav = torch.Tensor(onesample['waveform']).view(bsize,1,-1,1).cuda()
        fmri = torch.Tensor(onesample['fmri']).view(bsize,-1).cuda()

        print("Wave shape : {}".format(wav.shape))
        print("fmri shape : {}".format(fmri.shape))

        fmri_p = net(wav)
        print("Predicted fmri shape : {}".format(fmri_p.shape))
        print("CRASH TEST SUCCESSFUL")





### Main Training Loop 
startdate = datetime.now()

train_loss = []
val_loss = []
for epoch in tqdm(range(nbepoch)):
    train_loss.append(train(epoch,trainloader,net,optimizer,mseloss=mseloss))
    val_loss.append(test(epoch,valloader,net,optimizer,mseloss=mseloss))
    print("Train : {}, Val : {} ".format(train_loss[-1],val_loss[-1]))
    lr_sched.step(val_loss[-1])

    # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
    early_stopping(val_loss[-1], net)
    
    #print(np.argmax(net.maskattention.detach().cpu().numpy(),axis=0))
    if early_stopping.early_stop:
        print("Early stopping")
        break

test_loss = test(1,testloader,net,optimizer,mseloss=mseloss)
#print("Test Loss : {}".format(test_loss))

enddate = datetime.now()

if not os.path.isdir(destdir):
    os.mkdir(destdir)

## Reload best model 
net.load_state_dict(torch.load('checkpoint.pt'))

dt_string = enddate.strftime("%Y-%m-%d-%H-%M-%S")
str_bestmodel = os.path.join(destdir,"{}.pt".format(dt_string))
str_bestmodel_plot = os.path.join(destdir,"{}_{}.png".format(dt_string,fmrihidden))
str_bestmodel_nii = os.path.join(destdir,"{}_{}.nii.gz".format(dt_string,fmrihidden))

# Remove temp file 
os.remove('checkpoint.pt')

r2model = test_r2(testloader,net,mseloss)
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
            'training_time' : enddate - startdate,
            'nhidden' : fmrihidden,
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
