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


### define DataSet for one video (to be iterated on all videos)

class AudioToEmbeddings(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, videofile,samplerate = 12000):
        """
        Args:
            videofile (string): Path to the mkv file of a video.
        """
        self.wavfile = (videofile[:-4] + '_subsampl.wav')
        self.npzfile = (videofile[:-4] + '_fm_proba.npz')

        self.sample_rate = samplerate


        ### fetch the proba for the three modalities 

        self.places_proba = np.load(self.npzfile)['places_proba']
        self.im_proba = np.load(self.npzfile)['im_proba']
        self.audioset_proba = np.load(self.npzfile)['audioset_proba']
        self.dur = np.load(self.npzfile)['dur']
        self.onsets = np.load(self.npzfile)['onsets']
        
        #### Check if audio file exists
        if os.path.isfile(self.wavfile) is False:

            #### If not, generate it and put it at the same place than the video file , as a wav, with the same name
            #### use this following audio file to generate predictions on sound 

            print('wav file does not exist, converting from {videofile}...'.format(videofile=videofile))

            convert_Audio(videofile, self.wavfile)

        wav,native_sr = librosa.core.load(self.wavfile,duration=2,sr=None)

        
        if int(native_sr)!=int(self.sample_rate):
            print("Native Sampling rate is {}".format(native_sr))

            print('Resampling to {sr} Hz'.format(sr=self.sample_rate))

            wav,_ = librosa.core.load(self.wavfile, sr=self.sample_rate, mono=True)
            soundfile.write(self.wavfile,wav,self.sample_rate)

    def __len__(self):
        return len(self.onsets)

    def __getitem__(self, idx):
        offset = idx * self.dur

        (waveform, _) = librosa.core.load(self.wavfile, sr=self.sample_rate, mono=True,offset=offset,duration=self.dur)

        sample = {'waveform':(waveform),'places':(self.places_proba[idx]),
            'audioset':(self.audioset_proba[idx]),'imagenet':(self.im_proba[idx])}

        
        return (sample)



#testvid = '/home/nfarrugi/git/neuromod/cneuromod/movie10/stimuli/life/life1_seg01.mkv'

trainsets = []
testsets= []
valsets = []

path = '/media/brain/Elec_HD/cneuromod/movie10/stimuli/'
for root, dirs, files in os.walk(path, topdown=False):
   for name in files:
       if name[-3:] == 'mkv':
           currentvid = os.path.join(root, name)
           print(currentvid)
           dataset = AudioToEmbeddings(currentvid)
           trainsets.append(torch.utils.data.Subset(dataset, range(400)))
           valsets.append(torch.utils.data.Subset(dataset, range(400,500)))
           testsets.append(torch.utils.data.Subset(dataset, range(500,599)))


trainset = torch.utils.data.ConcatDataset(trainsets)
valset = torch.utils.data.ConcatDataset(valsets)
testset = torch.utils.data.ConcatDataset(testsets)

trainloader = DataLoader(trainset,batch_size=64,shuffle=True)
valloader = DataLoader(valset,batch_size=64)
testloader = DataLoader(testset,batch_size=64)

tau = 0.1 


def train_kl(epoch,trainloader):

    running_loss = 0
    net.train()

    for batch_idx, (onesample) in enumerate(trainloader):


        optimizer.zero_grad()
        bsize = onesample['waveform'].shape[0]

        

        # load data
        wav = torch.Tensor(onesample['waveform']).view(bsize,1,-1,1).cuda()
        places = torch.Tensor(onesample['places']).view(bsize,-1,1,1).cuda()
        audioset = torch.Tensor(onesample['audioset']).view(bsize,-1,1,1).cuda()
        imnet = torch.Tensor(onesample['imagenet']).view(bsize,-1,1,1).cuda()

        # Forward pass
        obj_p,scene_p,audio_p = net(wav)

        # Calculate loss

        loss_imagenet = kl_im(F.log_softmax(obj_p,1),imnet)
        loss_audioset = kl_audio(F.log_softmax(audio_p,1),audioset)
        loss_places = kl_places(F.log_softmax(scene_p,1),places)
        
        loss = loss_audioset + loss_imagenet + loss_places


        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()

        
    return running_loss/batch_idx


def test_kl(epoch,testloader):

    running_loss = 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (onesample) in enumerate(testloader):

            bsize = onesample['waveform'].shape[0]

            # load data
            wav = torch.Tensor(onesample['waveform']).view(bsize,1,-1,1).cuda()
            places = torch.Tensor(onesample['places']).view(bsize,-1,1,1).cuda()
            audioset = torch.Tensor(onesample['audioset']).view(bsize,-1,1,1).cuda()
            imnet = torch.Tensor(onesample['imagenet']).view(bsize,-1,1,1).cuda()

            # Forward pass
            obj_p,scene_p,audio_p = net(wav)

            # Calculate loss
            
            loss_imagenet = kl_im(F.log_softmax(obj_p,1),imnet)
            loss_audioset = kl_audio(F.log_softmax(audio_p,1),audioset)
            loss_places = kl_places(F.log_softmax(scene_p,1),places)
        
            loss = loss_audioset + loss_imagenet + loss_places

            running_loss += loss.item()
            
    return running_loss/batch_idx




net = WaveformCNN(nfeat=8)
net = net.cuda()
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)

#optimizer = torch.optim.Adam(net.parameters())

kl_im = nn.KLDivLoss(reduction='batchmean')
kl_audio = nn.KLDivLoss(reduction='batchmean')
kl_places = nn.KLDivLoss(reduction='batchmean')

nbepoch = 50
train_loss = []
val_loss = []
for epoch in tqdm(range(nbepoch)):
    train_loss.append(train_kl(epoch,trainloader))
    val_loss.append(test_kl(epoch,valloader))
    #print("Train : {}, Val : {} ".format(train_loss,val_loss))

print("Test Loss : {}".format(test_kl(1,testloader)))

plt.plot(range(nbepoch),train_loss)
plt.plot(range(nbepoch),val_loss)
plt.show()





