import torch, warnings
import torch.nn as nn
from nistats import hemodynamic_models
import numpy as np

class WaveformCNN8(nn.Module):
    def __init__(self,nfeat=16,ninputfilters=16,do_encoding_fmri=False,nroi=210,fmrihidden=1000,nroi_attention=None):
        super(WaveformCNN8, self).__init__()
        raise("DEPRECATED")
        ## Hyper parameters of the main branch
        self.nfeat = nfeat
        self.ninputfilters = 16

        ### Hyper parameters of the fmri encoding model 
        self.do_encoding_fmri = do_encoding_fmri
        self.nroi = nroi
        self.fmrihidden = fmrihidden

        ### Define all modules

        self.define_module()

        ### Define Output attention selection parameter
        if nroi_attention is not None:
            self.maskattention = torch.nn.Parameter(torch.rand(nroi,nroi_attention))
        else:
            self.maskattention = None
        
        
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
    def __init__(self,nfeat=16,ninputfilters=16,do_encoding_fmri=False,nroi=210,fmrihidden=1000,nroi_attention=None):
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
        ### Define Output attention selection parameter
        if nroi_attention is not None:
            self.maskattention = torch.nn.Parameter(torch.rand(nroi,nroi_attention))
        else:
            self.maskattention = None
        
        
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


class SoundNet8_pytorch(nn.Module):
    def __init__(self):
        super(SoundNet8_pytorch, self).__init__()
        
        self.define_module()
        
    def define_module(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16, (64,1), (2,1), (32,0), bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((8,1), (8,1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (32,1), (2,1), (16,0), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((8,1),(8,1))
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
        ) # difference here (0.24751323, 0.2474), padding error has beed debuged
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
        self.conv8 = nn.Sequential(
            nn.Conv2d(1024, 1000, (8,1), (2,1), (0,0), bias=True),
        ) 
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(1024, 401, (8,1), (2,1), (0,0), bias=True)
        )

    def forward(self, x):
        for net in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]:
            x = net(x)
            
        ### "Truncated" soundnet to only extract conv7 

        #object_pred = self.conv8(x)
        #scene_pred = self.conv8_2(x) 
        return x


class SoundNetEncoding(nn.Module):
    def __init__(self,pytorch_param_path,nroi=210,fmrihidden=1000,nroi_attention=None, hrf_model=None, oversampling = 16, tr = 1.49, audiopad = 0):
        super(SoundNetEncoding, self).__init__()

        self.soundnet = SoundNet8_pytorch()
        self.fmrihidden = fmrihidden
        self.nroi = nroi

        print("Loading SoundNet weights...")
        # load pretrained weights of original soundnet model
        self.soundnet.load_state_dict(torch.load(pytorch_param_path))

        #freeze the parameters of soundNet
        for param in self.soundnet.parameters():
            param.requires_grad = False

        print("Pretrained model loaded")

        self.gpool = nn.AdaptiveAvgPool2d((1,1)) # Global average pooling

        self.encoding_fmri = nn.Sequential(                
                nn.Linear(1024,self.fmrihidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.fmrihidden,self.nroi)
            )
            
        if hrf_model is not None : 
            self.hrf_model = hrf_model
            self.oversampling = oversampling
            self.audiopad = audiopad
            self.tr = tr
        else :
            self.hrf_model=None

    def forward(self, x, onsets, durations):
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            emb = self.soundnet(x)
            emb = self.gpool(emb)
            emb = emb.view(-1,1024)

            if self.hrf_model is not None :
                fvs = emb.cpu().numpy()

                index_zeros = np.where(onsets == 0)
                if len(index_zeros[0]) > 0 and index_zeros[0][0] != 0:
                    n_onsets = np.split(onsets, index_zeros[0]) 
                    n_durations = np.split(durations, index_zeros[0])
                    fvs = np.split(fvs, index_zeros[0])
                else:
                    n_onsets = [onsets] 
                    n_durations = [durations]
                    fvs = [fvs]

                all_fv = np.array([]).reshape(emb.shape[1], 0)
                for onset, duration, fv in zip(n_onsets, n_durations, fvs):
                    fv_temp = []
                    frame_times = onset.numpy()

                    for amplitude in fv.T:
                        exp_conditions = (onset, duration, amplitude)
                        signal, _ = hemodynamic_models.compute_regressor(exp_conditions, 
                                    self.hrf_model, frame_times, oversampling=self.oversampling, min_onset=0)
                        fv_temp.append(signal)
                    b = np.squeeze(np.stack(fv_temp))
                    all_fv = np.concatenate((all_fv, b),axis=1)

                emb = np.squeeze(np.stack(all_fv)).T
                emb = torch.from_numpy(emb).float().cuda()

        out = self.encoding_fmri(emb)
        
        return out

class SoundNetEncoding_conv(nn.Module):
    def __init__(self,pytorch_param_path,nroi=210,fmrihidden=1000,nroi_attention=None, hrf_model=None, oversampling = 16, tr = 1.49, audiopad = 0,transfer=True,preload=True):
        super(SoundNetEncoding_conv, self).__init__()

        self.soundnet = SoundNet8_pytorch()
        self.fmrihidden = fmrihidden
        self.nroi = nroi

        if preload:
            print("Loading SoundNet weights...")
            # load pretrained weights of original soundnet model
            self.soundnet.load_state_dict(torch.load(pytorch_param_path))
            print("Pretrained model loaded")
            if transfer:
                #freeze the parameters of soundNet
                print("Transfer learning - backbone is fixed")
                for param in self.soundnet.parameters():
                    param.requires_grad = False
            else:
                print("Finetuning : backbone will be optimized")

        self.encoding_fmri = nn.Sequential(                
                nn.Conv2d(1024,self.nroi,kernel_size=(1,1)),
                #nn.ReLU(inplace=True),
                #nn.Conv2d(self.fmrihidden,self.nroi,kernel_size=(1,1)),

            )

        if nroi_attention is not None:
            self.maskattention = torch.nn.Parameter(torch.rand(nroi,nroi_attention))
        else:
            self.maskattention = None
        
            
        if hrf_model is not None : 
            self.hrf_model = hrf_model
            self.oversampling = oversampling
            self.audiopad = audiopad
            self.tr = tr
        else :
            self.hrf_model=None

    def forward(self, x):
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            emb = self.soundnet(x)

        out = self.encoding_fmri(emb)
        
        return out


class SoundNetEncoding_conv_2(nn.Module):
    def __init__(self,pytorch_param_path,nroi=210,fmrihidden=1000,nroi_attention=None, hrf_model=None, oversampling = 16, tr = 1.49, audiopad = 0,transfer=True,preload=True):
        super(SoundNetEncoding_conv_2, self).__init__()

        self.soundnet = SoundNet8_pytorch()
        self.fmrihidden = fmrihidden
        self.nroi = nroi

        if preload:
            print("Loading SoundNet weights...")
            # load pretrained weights of original soundnet model
            self.soundnet.load_state_dict(torch.load(pytorch_param_path))
            print("Pretrained model loaded")
            if transfer:
                #freeze the parameters of soundNet
                print("Transfer learning - backbone is fixed")
                for param in self.soundnet.parameters():
                    param.requires_grad = False
            else:
                print("Finetuning : backbone will be optimized")

        self.encoding_fmri = nn.Sequential(                
                nn.Conv2d(1024,self.fmrihidden,kernel_size=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.fmrihidden,self.nroi,kernel_size=(1,1)),

            )

        if nroi_attention is not None:
            self.maskattention = torch.nn.Parameter(torch.rand(nroi,nroi_attention))
        else:
            self.maskattention = None
        
            
        if hrf_model is not None : 
            self.hrf_model = hrf_model
            self.oversampling = oversampling
            self.audiopad = audiopad
            self.tr = tr
        else :
            self.hrf_model=None

    def forward(self, x):
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            emb = self.soundnet(x)

        out = self.encoding_fmri(emb)
        
        return out


class SoundNetEncoding_conv_3(nn.Module):
    def __init__(self,pytorch_param_path,nroi=210,fmrihidden=1000,nroi_attention=None, hrf_model=None, oversampling = 16, tr = 1.49, audiopad = 0,transfer=True,preload=True):
        super(SoundNetEncoding_conv_3, self).__init__()

        self.soundnet = SoundNet8_pytorch()
        self.fmrihidden = fmrihidden
        self.nroi = nroi

        if preload:
            print("Loading SoundNet weights...")
            # load pretrained weights of original soundnet model
            self.soundnet.load_state_dict(torch.load(pytorch_param_path))
            print("Pretrained model loaded")
            if transfer:
                #freeze the parameters of soundNet
                print("Transfer learning - backbone is fixed")
                for param in self.soundnet.parameters():
                    param.requires_grad = False
            else:
                print("Finetuning : backbone will be optimized")

            

        self.encoding_fmri = nn.Sequential(                
                nn.Conv2d(1024,2*self.fmrihidden,kernel_size=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(2*self.fmrihidden,self.fmrihidden,kernel_size=(1,1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.fmrihidden,self.nroi,kernel_size=(1,1)),

            )

        if nroi_attention is not None:
            self.maskattention = torch.nn.Parameter(torch.rand(nroi,nroi_attention))
        else:
            self.maskattention = None
        
            
        if hrf_model is not None : 
            self.hrf_model = hrf_model
            self.oversampling = oversampling
            self.audiopad = audiopad
            self.tr = tr
        else :
            self.hrf_model=None

    def forward(self, x):
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            emb = self.soundnet(x)

        out = self.encoding_fmri(emb)
        
        return out