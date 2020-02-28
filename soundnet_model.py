import torch
import torch.nn as nn

class WaveformCNN8(nn.Module):
    def __init__(self,nfeat=16,ninputfilters=16,do_encoding_fmri=False,nroi=210,fmrihidden=1000):
        super(WaveformCNN8, self).__init__()

        ## Hyper parameters of the main branch
        self.nfeat = nfeat
        self.ninputfilters = 16

        ### Hyper parameters of the fmri encoding model 
        self.do_encoding_fmri = do_encoding_fmri
        self.nroi = nroi
        self.fmrihidden = fmrihidden

        ### Define all modules

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
        if self.do_encoding_fmri:
            self.encoding_fmri = nn.Sequential(
                nn.Conv2d(64*self.nfeat,self.fmrihidden,(10,1),bias=True),
                nn.ReLU(inplace=True)
            )
            self.outputroi = nn.Linear(self.fmrihidden,self.nroi)



    def forward(self, x):
        for net in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]:
            x = net(x)
        object_emb = self.object_emb(x)
        scene_emb = self.scene_emb(x) 
        audiotag_emb = self.audiotag_emb(x)
        if self.do_encoding_fmri:
            x=self.encoding_fmri(x)
            x=self.outputroi(x.view(-1,1000))
        return object_emb, scene_emb, audiotag_emb,x



class WaveformCNN5(nn.Module):
    def __init__(self,nfeat=16,ninputfilters=16,do_encoding_fmri=False,nroi=210,fmrihidden=1000):
        super(WaveformCNN5, self).__init__()

        ## Hyper parameters of the main branch
        self.nfeat = nfeat
        self.ninputfilters = 16

        ### Hyper parameters of the fmri encoding model 
        self.do_encoding_fmri = do_encoding_fmri
        self.nroi = nroi
        self.fmrihidden = fmrihidden

        ### Define all modules

        self.define_module()
        
        
    def define_module(self):

        #The hyperparameters of this network have been set for 12 kHz, 1.49 second long waveforms. 

        self.conv1 = nn.Sequential(
            nn.Conv2d(1,self.ninputfilters, (32,1), (4,1), (32,0), bias=True),
            nn.BatchNorm2d(self.ninputfilters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1), (2,1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.ninputfilters, 32*self.nfeat, (16,1), (4,1), (16,0), bias=True),
            nn.BatchNorm2d(32*self.nfeat),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1),(2,1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32*self.nfeat, 64*self.nfeat, (8,1), (2,1), (8,0), bias=True),
            nn.BatchNorm2d(64*self.nfeat),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64*self.nfeat, 128*self.nfeat, (4,1), (4,1), (4,0), bias=True),
            nn.BatchNorm2d(128*self.nfeat),
            nn.ReLU(inplace=True)
        )

        self.gpool = nn.AdaptiveAvgPool2d((1,1)) # Global average pooling
        
        self.object_emb = nn.Sequential(
            nn.Linear(128*self.nfeat, 1000),
        ) 
        self.scene_emb = nn.Sequential(
             nn.Linear(128*self.nfeat, 365),
        )
        self.audiotag_emb = nn.Sequential(
             nn.Linear(128*self.nfeat, 527),
        )
        if self.do_encoding_fmri:
            self.encoding_fmri = nn.Sequential(
                nn.Linear(128*self.nfeat,self.fmrihidden),
                nn.ReLU(inplace=True)
            )
            self.outputroi = nn.Linear(self.fmrihidden,self.nroi)


    def forward(self, x):
        for net in [self.conv1, self.conv2, self.conv3, self.conv4]:
            x = net(x)
            
        x =self.gpool(x)
        
        object_emb = self.object_emb(x.view(-1,1,128*self.nfeat))
        scene_emb = self.scene_emb(x.view(-1,1,128*self.nfeat)) 
        audiotag_emb = self.audiotag_emb(x.view(-1,1,128*self.nfeat))
        if self.do_encoding_fmri:
            x=self.encoding_fmri(x.view(-1,1,128*self.nfeat))
            x=self.outputroi(x)
            
        return object_emb, scene_emb, audiotag_emb,x


