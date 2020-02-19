## Author : Nicolas Farrugia, February 2020
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
import librosa
import soundfile
from soundnet_model import SmallerWaveCNN


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



testvid = '/home/nfarrugi/git/neuromod/cneuromod/movie10/stimuli/life/life1_seg01.mkv'

dataset = AudioToEmbeddings(testvid)

dataloader = DataLoader(dataset,batch_size=64,shuffle=True)



#### Simple Test case just to check the shapes

if False:
    onesample = dataset.__getitem__(25)

    wav = torch.Tensor(onesample['waveform']).view(1,1,-1,1)

    places = torch.Tensor(onesample['places']).view(1,-1,1,1)

    audioset = torch.Tensor(onesample['audioset']).view(1,-1,1,1)

    imnet = torch.Tensor(onesample['imagenet']).view(1,-1,1,1)

    model =SmallerWaveCNN()

    obj_p,scene_p,audio_p = model(wav)
    print(obj_p.shape,scene_p.shape,audio_p.shape)
    print(places.shape,audioset.shape,imnet.shape)



    kl = nn.KLDivLoss(reduction='batchmean')
    print(kl(audioset,audio_p))


def train_kl(epochs,net,optimizer,dataloader):


    kl_im = nn.KLDivLoss(reduction='batchmean')
    kl_audio = nn.KLDivLoss(reduction='batchmean')
    kl_places = nn.KLDivLoss(reduction='batchmean')
    running_loss = 0

    for batch_idx, (onesample) in enumerate(dataloader):

        ## zero grad
        optimizer.zero_grad()
        bsize = onesample['waveform'].shape[0]

        # load data
        wav = torch.Tensor(onesample['waveform']).view(bsize,1,-1,1)
        places = torch.Tensor(onesample['places']).view(bsize,-1,1,1)
        audioset = torch.Tensor(onesample['audioset']).view(bsize,-1,1,1)
        imnet = torch.Tensor(onesample['imagenet']).view(bsize,-1,1,1)

        # Forward pass
        obj_p,scene_p,audio_p = net(wav)

        # Calculate loss

        loss_imagenet = kl_im(imnet,obj_p)
        loss_audioset = kl_audio(places,scene_p)
        loss_places = kl_places(audioset,audio_p)
        
        loss = loss_audioset + loss_imagenet + loss_places

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss




model = SmallerWaveCNN()

optimizer = torch.optim.Adam(model.parameters())


for epoch in range(10):
    train_loss = train_kl(epoch,model,optimizer,dataloader)
    print(train_loss)







