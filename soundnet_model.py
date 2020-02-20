import torch
import torch.nn as nn

class WaveformCNN(nn.Module):
    def __init__(self,nfeat=16,ninputfilters=16):
        super(WaveformCNN, self).__init__()
        self.nfeat = nfeat
        self.ninputfilters = 16
        self.define_module()
        
        
    def define_module(self):

        #The hyperparameters of this network have been set for 12 kHz, 1.49 second long waveforms. 

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,self.ninputfilters, (64,1), (2,1), (32,0), bias=True),
            nn.BatchNorm2d(self.ninputfilters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1), (2,1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.ninputfilters, 2*self.nfeat, (32,1), (2,1), (16,0), bias=True),
            nn.BatchNorm2d(2*self.nfeat),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1),(2,1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*self.nfeat, 4*self.nfeat, (16,1), (2,1), (8,0), bias=True),
            nn.BatchNorm2d(4*self.nfeat),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4*self.nfeat, 8*self.nfeat, (8,1), (2,1), (4,0), bias=True),
            nn.BatchNorm2d(8*self.nfeat),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(8*self.nfeat, 16*self.nfeat, (4,1),(2,1),(2,0), bias=True),
            nn.BatchNorm2d(16*self.nfeat),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,1),(4,1))
        ) 
        self.conv6 = nn.Sequential(
            nn.Conv2d(16*self.nfeat, 32*self.nfeat, (4,1), (2,1), (2,0), bias=True),
            nn.BatchNorm2d(32*self.nfeat),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(32*self.nfeat, 64*self.nfeat, (4,1), (2,1), (2,0), bias=True),
            nn.BatchNorm2d(64*self.nfeat),
            nn.ReLU(inplace=True)
        )
        self.object_emb = nn.Sequential(
            nn.Conv2d(64*self.nfeat, 1000, (10,1), bias=True),
        ) 
        self.scene_emb = nn.Sequential(
            nn.Conv2d(64*self.nfeat, 365, (10,1), bias=True)
        )
        self.audiotag_emb = nn.Sequential(
            nn.Conv2d(64*self.nfeat, 527, (10,1), bias=True)
        )


    def forward(self, x):
        for net in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]:
            x = net(x)
        object_emb = self.object_emb(x)
        scene_emb = self.scene_emb(x) 
        audiotag_emb = self.audiotag_emb(x)
        return object_emb, scene_emb, audiotag_emb




class SmallerWaveCNN(nn.Module):
    def __init__(self):
        super(SmallerWaveCNN, self).__init__()
        
        self.define_module()
        
    def define_module(self):

        #The hyperparameters of this network have been set for 12 kHz, 1.49 second long waveforms. 

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16, (64,1), (2,1), (32,0), bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1), (2,1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (32,1), (2,1), (16,0), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1),(2,1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (16,1), (2,1), (8,0), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, (8,1), (2,1), (4,0), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, (4,1),(2,1),(2,0), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,1),(4,1))
        ) 
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, (4,1), (2,1), (2,0), bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, (4,1), (2,1), (2,0), bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.object_emb = nn.Sequential(
            nn.Conv2d(1024, 1000, (10,1), bias=True),
        ) 
        self.scene_emb = nn.Sequential(
            nn.Conv2d(1024, 365, (10,1), bias=True)
        )
        self.audiotag_emb = nn.Sequential(
            nn.Conv2d(1024, 527, (10,1), bias=True)
        )


    def forward(self, x):
        for net in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]:
            x = net(x)
        object_emb = self.object_emb(x)
        scene_emb = self.scene_emb(x) 
        audiotag_emb = self.audiotag_emb(x)
        return object_emb, scene_emb, audiotag_emb




class SoundNet8_pytorch(nn.Module):
    def __init__(self):
        super(SoundNet8_pytorch, self).__init__()
        
        self.define_module()
        
    def define_module(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16, (64,1), (2,1), (32,0), bias=True),
            #nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((8,1), (8,1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (32,1), (2,1), (16,0), bias=True),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((8,1),(8,1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (16,1), (2,1), (8,0), bias=True),
            #nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, (8,1), (2,1), (4,0), bias=True),
            #nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, (4,1),(2,1),(2,0), bias=True),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,1),(4,1))
        ) # difference here (0.24751323, 0.2474), padding error has beed debuged
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, (4,1), (2,1), (2,0), bias=True),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, (4,1), (2,1), (2,0), bias=True),
            #nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.object_emb = nn.Sequential(
            nn.Conv2d(1024, 512, (8,1), (2,1), (0,0), bias=True),
        ) 
        self.scene_emb = nn.Sequential(
            nn.Conv2d(1024, 512, (8,1), (2,1), (0,0), bias=True)
        )
        self.audiotag_emb = nn.Sequential(
            nn.Conv2d(1024, 512, (8,1), (2,1), (0,0), bias=True)
        )


    def forward(self, x):
        for net in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]:
            x = net(x)
        object_emb = self.object_emb(x)
        scene_emb = self.scene_emb(x) 
        audiotag_emb = self.audiotag_emb(x)
        return object_emb, scene_emb, audiotag_emb
