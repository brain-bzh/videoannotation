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
from soundnet_model import SoundNetEncoding,SoundNetEncoding_conv,SoundNetEncoding_conv_2,SoundNetEncoding_conv_3
import argparse

parser = argparse.ArgumentParser(description='Neuromod Movie10 Distillation-transferred encoding model')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch', default=12, type=int, help='batch size (also corresponds to number of TR in a row)')
parser.add_argument('--epochs', default=5000, type=int, help='Maximum number of epochs')
parser.add_argument('--hidden', default=1000, type=int, help='Number of neurons for hidden layer in the encoding model (previous layer has 128*expansion fm)')
parser.add_argument('--model', default=0, type=int, help='Which model (0, 1 or 2)')
parser.add_argument('--scratch', action='store_true', help='Train from scratch (default False = load pretrained soundnet)')
parser.add_argument('--finetune', action='store_true', help='Fine tune instead of transfer (default False = transfer)')
parser.add_argument('--nroi_attention', default=None, type=int, help='number of regions to learn using outputattention')
parser.add_argument('--resume', default=None, type=str, help='Path to model checkpoint to resume training')
parser.add_argument('--delta', default=1e-1, type=float, help='MSE penalty')
parser.add_argument('--epsilon', default=1e-2, type=float, help='Ortho penalty for attention')
parser.add_argument('--movie', default='/home/nfarrugi/git/neuromod/cneuromod/movie10/stimuli', type=str, help='Path to the movie directory')
parser.add_argument('--subject', default='/home/nfarrugi/movie10_parc/sub-01', type=str, help='Path to the subject parcellation directory')
parser.add_argument('--audiopad', default = 0, type=int, help='size of audio padding to take in account for one audio unit learned')
parser.add_argument('--save_path', default='.', type=str, help='path to results')
parser.add_argument('--hrf_model', default=None, type=str, help='hrf model to compute the regressors of the hemodynamic response')
args = parser.parse_args()

from train_utils import construct_dataloader, construct_iter_dataloader

from train_utils import train,test

from pytorchtools import EarlyStopping
from datetime import datetime

from nilearn.plotting import plot_stat_map

from nilearn.regions import signals_to_img_labels

from eval_utils import test_r2

#mistroifile = '/home/nfarrugi/git/GSP/MIST_ROI.nii.gz'
mistroifile = '/home/maelle/Database/MIST_parcellation/Parcellations/MIST_ROI.nii.gz'


mv_path = args.movie
sub_path = args.subject
audiopad = args.audiopad
hrf_model = args.hrf_model
#trainloader, valloader, testloader, dataset = construct_dataloader(mv_path, sub_path, audiopad,bsize=args.batch)
trainloader, valloader, testloader, dataset = construct_iter_dataloader(mv_path, sub_path,bsize=args.batch)

nroi_attention = args.nroi_attention
fmrihidden = args.hidden

print(args)

models = [SoundNetEncoding_conv, SoundNetEncoding_conv_2, SoundNetEncoding_conv_3]

chosen_model = models[args.model]
destdir = args.save_path
#destdir = os.path.join(save_path, 'cp_model_{}'.format(args.model))
#if not os.path.isdir(destdir):
#    os.mkdir(destdir)

### Model Setup
if args.resume is not None:
    print("Reloading model {}".format(args.resume))
    old_dict = torch.load(args.resume)
    net = old_dict['model']
    net.load_state_dict(old_dict['net'])
else:
    net = chosen_model(pytorch_param_path='./sound8.pth',fmrihidden=fmrihidden,nroi_attention=nroi_attention, 
                        hrf_model=hrf_model,transfer=not(args.finetune),preload=not(args.scratch))
del models
net = net.cuda()
mseloss = nn.MSELoss(reduction='sum')

### Optimizer and Schedulers
#optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr = args.lr)
lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2,patience=5,threshold=1e-2,cooldown=2)

early_stopping = EarlyStopping(patience=10, verbose=True,delta=1e-6)

nbepoch = args.epochs

### Main Training Loop 
startdate = datetime.now()

train_loss = []
train_r2_max = []
train_r2_mean = []
val_loss = []
val_r2_max = []
val_r2_mean = []
try:
    for epoch in tqdm(range(nbepoch)):
        t_l, t_r2 = train(epoch,trainloader,net,optimizer,mseloss=mseloss,delta=args.delta,epsilon=args.epsilon)
        train_loss.append(t_l)
        train_r2_max.append(max(t_r2))
        train_r2_mean.append(np.mean(t_r2))

        v_l, v_r2 = test(epoch,valloader,net,optimizer,mseloss=mseloss,delta=args.delta,epsilon=args.epsilon)
        val_loss.append(v_l)
        val_r2_max.append(max(v_r2))
        val_r2_mean.append(np.mean(v_r2))
        print("Train Loss {} Train Mean R2 :  {} Train Max R2 : {}, Val Loss {} Val Mean R2:  {} Val Max R2 : {} ".format(train_loss[-1],train_r2_mean[-1],train_r2_max[-1],val_loss[-1],val_r2_mean[-1],val_r2_max[-1]))
        lr_sched.step(val_loss[-1])

        # early_stopping needs the R2 mean to check if it has increased, 
        # and if it has, it will make a checkpoint of the current model
        r2_forEL = -(val_r2_max[-1])
        early_stopping(r2_forEL, net)
        if net.maskattention is not None:
            print(np.argmax(net.maskattention.detach().cpu().numpy(),axis=0))
        if early_stopping.early_stop:
            print("Early stopping")
            break

except KeyboardInterrupt:
    print("Interrupted by user")

test_loss = test(1,testloader,net,optimizer,mseloss=mseloss)
#print("Test Loss : {}".format(test_loss))

enddate = datetime.now()



## Reload best model 
net.load_state_dict(torch.load('checkpoint.pt'))

dt_string = enddate.strftime("%Y-%m-%d-%H-%M-%S")
str_bestmodel = os.path.join(destdir,"{}_{}_{}_{}_{}.pt".format(dt_string, args.delta, args.epsilon,args.lr,args.batch))
str_bestmodel_plot = os.path.join(destdir,"{}_{}_{}_{}_{}.png".format(dt_string,args.delta, args.epsilon, args.lr,args.batch))
str_bestmodel_nii = os.path.join(destdir,"{}_{}_{}_{}_{}.nii.gz".format(dt_string,args.delta, args.epsilon, args.lr,args.batch))

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
            'train_r2_max' : train_r2_max,
            'train_r2_mean' : train_r2_mean,
            'val_loss' : val_loss,
            'val_r2_max' : val_r2_max,
            'val_r2_mean' : val_r2_mean,
            'test_loss' : test_loss,
            'r2' : r2model,
            'r2max' : r2model.max(),
            'r2mean' : r2model.mean(),
            'training_time' : enddate - startdate,
            'nhidden' : fmrihidden,
            'model' : net
        }


### Plot the loss figure
f = plt.figure(figsize=(20,40))

ax = plt.subplot(4,1,2)

plt.plot(state['train_loss'][1:])
plt.plot(state['val_loss'][1:])
plt.legend(['Train','Val'])
plt.title("loss evolution => Mean test R^2=${}, Max test R^2={}, for model {}, batchsize ={} and {} hidden neurons".format(r2model.mean(),r2model.max(), args.model, args.batch, fmrihidden))

### Mean R2 evolution during training
ax = plt.subplot(4,1,3)

plt.plot(state['train_r2_mean'][1:])
plt.plot(state['val_r2_mean'][1:])
plt.legend(['Train','Val'])
plt.title("Mean R^2 evolution for model {}, batchsize ={} and {} hidden neurons".format(args.model, args.batch, fmrihidden))

### Max R2 evolution during training
ax = plt.subplot(4,1,4)

plt.plot(state['train_r2_max'][1:])
plt.plot(state['val_r2_max'][1:])
plt.legend(['Train','Val'])
plt.title("Max R^2 evolution for model {}, batchsize ={} and {} hidden neurons".format(args.model, args.batch, fmrihidden))

### R2 figure 
r2_img = signals_to_img_labels(r2model.reshape(1,-1),mistroifile)

ax = plt.subplot(4,1,1)

plot_stat_map(r2_img,display_mode='z',cut_coords=8,figure=f,axes=ax)
f.savefig(str_bestmodel_plot)
r2_img.to_filename(str_bestmodel_nii)
plt.close()

torch.save(state, str_bestmodel)

### Plot the Mask attention figure 
if net.maskattention is not None:
    str_bestmodel_mask = os.path.join(destdir,"{}_{}_{}_{}_{}_mask.png".format(dt_string,args.delta, args.epsilon, args.lr,args.batch))

    mask_att = net.maskattention.detach().cpu().numpy().T
    #print(mask_att.shape)
    f= plt.figure(figsize=(20,40))

    for i,curmask in enumerate(mask_att):
        mask_att_img = signals_to_img_labels(curmask.reshape(1,-1),mistroifile)

        ax = plt.subplot(mask_att.shape[0],1,i+1)


        plot_stat_map(mask_att_img,display_mode='z',cut_coords=8,figure=f,axes=ax,threshold='auto')

    f.savefig(str_bestmodel_mask)