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



### define DataSet for one video (to be iterated on all videos)

class AudioToEmbeddings(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, videofile,samplerate = 12000):
        """
        Args:
            videofile (string): Path to the mkv file of a video.
        """
        self.wavfile = (videofile[:-4] + '_subsampl.wav')
        self.npzfile = (videofile[:-4] + '_fm.npz')

        self.sample_rate = samplerate


        ### fetch the feature vectors for the three modalities 

        self.places_fm = np.load(self.npzfile)['places_fm']
        self.im_fm = np.load(self.npzfile)['im_fm']
        self.audioset_fm = np.load(self.npzfile)['audioset_fm']
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

        sample = {'waveform':(waveform),'places':(self.places_fm[idx]),'imagenet':(self.im_fm)}

        
        return transforms.ToTensor(sample)



testvid = '/home/nfarrugi/git/neuromod/cneuromod/movie10/stimuli/wolf_of_wall_street/the_wolf_of_wall_street_seg01.mkv'

testdataset = AudioToEmbeddings(testvid)

print(len(testdataset))

dataloader = DataLoader(testdataset,batch_size=16,shuffle=True)