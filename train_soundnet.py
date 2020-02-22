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
from train_utils import train_kl,test_kl,AudioToEmbeddings,trainloader,valloader,testloader
from pytorchtools import EarlyStopping
from datetime import datetime

nfeat = 2
ninputfilters = 8


for ninputfilters in [8,16]:
    for nfeat in [int(ninputfilters/2),ninputfilters,2*ninputfilters,4*ninputfilters]:

        ### Model Setup
        net = WaveformCNN(nfeat=nfeat,ninputfilters=ninputfilters)
        net = net.cuda()
        kl_im = nn.KLDivLoss(reduction='batchmean')
        kl_audio = nn.KLDivLoss(reduction='batchmean')
        kl_places = nn.KLDivLoss(reduction='batchmean')

        ### Optimizer and Schedulers
        optimizer = torch.optim.SGD(net.parameters(),lr=0.001)
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2,patience=4,threshold=1e-4)
        #optimizer = torch.optim.Adam(net.parameters())

        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=8, verbose=True)
        nbepoch = 5000

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

                print(wav.shape)
                obj_p,scene_p,audio_p = net(wav)
                print(obj_p.shape,scene_p.shape,audio_p.shape)
                print(imnet.shape,places.shape,audioset.shape)

        ### Main Training Loop 
        startdate = datetime.now()

        train_loss = []
        val_loss = []
        for epoch in tqdm(range(nbepoch)):
            train_loss.append(train_kl(epoch,trainloader,net,optimizer,kl_im,kl_audio,kl_places))
            val_loss.append(test_kl(epoch,valloader,net,optimizer,kl_im,kl_audio,kl_places))
            #print("Train : {}, Val : {} ".format(train_loss[-1],val_loss[-1]))
            lr_sched.step(val_loss[-1])

            # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss[-1], net)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

        test_loss = test_kl(1,testloader,net,optimizer,kl_im,kl_audio,kl_places)
        print("Test Loss : {}".format(test_loss))

        enddate = datetime.now()

        ## Reload best model 
        net.load_state_dict(torch.load('checkpoint.pt'))

        ## Prepare data structure for checkpoint
        state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'train_loss' : train_loss,
                    'val_loss' : val_loss,
                    'test_loss' : test_loss,
                    'nfeat' : nfeat,
                    'ninputfilters' : ninputfilters,
                    'model' : net
                }

        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')

        dt_string = enddate.strftime("%Y-%m-%d-%H-%M-%S")
        str_bestmodel = os.path.join('checkpoints',"{}.pt".format(dt_string))
        str_bestmodel_plot = os.path.join('checkpoints',"{}_{}_{}.png".format(dt_string,ninputfilters,nfeat))

        torch.save(state, str_bestmodel)

        # Remove temp file 
        os.remove('checkpoint.pt')

        ## Plot losses 
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.ylim([7,11])
        plt.savefig(str_bestmodel_plot)
        plt.close()