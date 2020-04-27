## Author : Nicolas Farrugia, Maëlle Freteault, February 2020
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
import glob
import os
from sklearn.metrics import r2_score
from random import sample,shuffle

### Utility function to fetch fMRI data
### Modified from Maëlle Freteault 
def fetchMRI(videofile,fmripath):
    ### isolate the mkv file (->filename) and the rest of the path (->videopath)
    videopath,filename = os.path.split(videofile)

    #formatting the name to correspond to mri run formatting
    name = filename.replace('_', '')
    if name.startswith('the'):
        name = name.replace('the', '', 1)
    if name.find('life') > -1 :
        name = name.replace('life1', 'life')

    name = name.replace('seg','_run-')

    ## Rename to match the parcellated filenames
    name = name.replace('.mkv','npz.npz')

    # list of all parcellated filenames 
    allnpzfiles = (os.listdir(fmripath))

    # match videofilename with parcellated files
    mriMatchs = []

    for curfile in allnpzfiles:
        if curfile[23:] == (name):
            #print(curfile[23:],(name))
            mriMatchs.append(curfile)    

    #in case of multiple run for 1 film segment
    association = {}
    keyList=[]
    name_seg = filename[:-4]

    if len(mriMatchs) > 1 :
        numSessions = []
        for run in mriMatchs :
            index_sess = run.find('ses-vid')
            numSessions.append(int(run[index_sess+7:index_sess+10]))
            
        if numSessions[0] < numSessions[1] : 
            association[name_seg+'_S1'] = [videofile, mriMatchs[0]]
            keyList.append(name_seg+'_S1')
            association[name_seg+'_S2'] = [videofile, mriMatchs[1]]
            keyList.append(name_seg+'_S2')
        else : 
            association[name_seg+'_S1'] = [videofile, mriMatchs[1]]
            keyList.append(name_seg+'_S1')
            association[name_seg+'_S2'] = [videofile, mriMatchs[0]]
            keyList.append(name_seg+'_S2')
    else :
        
        association[name_seg] = [videofile, mriMatchs[0]]
        keyList.append(name_seg)

    return association

### define DataSet for one video (to be iterated on all videos)

class AudioToEmbeddings(torch.utils.data.Dataset):
    """Dataset that enables to load an audiofile synchronized with fMRI data and ImageNet, Places and AudioSet embeddings for Neuromod movie10."""

    def __init__(self, videofile,samplerate = 12000,fmripath=None, audioPad=1):
        """
        Args:
            videofile (string): Path to the mkv file of a video.
        """
        self.wavfile = (videofile[:-4] + '_subsampl.wav')
        self.npzfile = (videofile[:-4] + '_fm_proba.npz')

        self.sample_rate = samplerate
        self.audioPad = audioPad

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

        ### Load the fmri data, if provided the path
        
        self.fmrifile = None

        if fmripath is not None:
            print('Finding corresponding MRI file(s)...')
            association = fetchMRI(videofile,fmripath)
            ## Currently this will only fetch the second session of the film if there are two sessions
            for _,item in association.items():
                self.fmrifile = os.path.join(fmripath,item[1])

                ### load npz file 

                self.fmri = torch.FloatTensor(np.load(self.fmrifile)['X'])

                ### Check shape relative to other data types
                if self.fmri.shape[0] != self.audioset_proba.shape[0]:
                    print("reshaping fmri and other data to minimum length of both")

                    min_len = min(self.fmri.shape[0],self.audioset_proba.shape[0])
                    self.fmri = self.fmri[:min_len,:]
                    self.audioset_proba = self.audioset_proba[:min_len,:]
                    self.im_proba = self.im_proba[:min_len,:]
                    self.places_proba = self.places_proba[:min_len,:]
                    self.onsets = self.onsets[:min_len]
        
        #### Based on Cneuromod design, create a design matrix to compute the Hrf regressors in the training        

    def __len__(self):
        return (self.fmri.shape[0] - 2*self.audioPad)

    def __getitem__(self, idx):
        try:
            offset = self.onsets[idx]
        except IndexError:
            raise(IndexError('Pb with sizes'))

        duration = self.dur*(2*self.audioPad+1)
        (waveform, _) = librosa.core.load(self.wavfile, sr=self.sample_rate, mono=True,offset=offset,duration=duration)

        sample = {'waveform':(waveform),'places':(self.places_proba[idx+self.audioPad]),
            'audioset':(self.audioset_proba[idx+self.audioPad]),'imagenet':(self.im_proba[idx]+self.audioPad)}

        if self.fmrifile is not None:
            sample['fmri'] = self.fmri[idx+self.audioPad]

        return (sample, offset, duration)


class AudioToEmbeddingsIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, videofile,samplerate = 12000,fmripath=None):
    

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

        ## Load just 2 seconds to check the sample rate 
        wav,native_sr = librosa.core.load(self.wavfile,duration=2,sr=None, mono=True)

        # Resample and save if necessary 
        if int(native_sr)!=int(self.sample_rate):
            print("Native Sampling rate is {}".format(native_sr))

            print('Resampling to {sr} Hz'.format(sr=self.sample_rate))

            wav,_ = librosa.core.load(self.wavfile, sr=self.sample_rate, mono=True)
            soundfile.write(self.wavfile,wav,self.sample_rate)

        

        ### Load the fmri data, if provided the path
        # This will potentially modify the onsets
        
        self.fmrifile = None

        if fmripath is not None:
            #print('Finding corresponding MRI file(s)...')
            association = fetchMRI(videofile,fmripath)
            ## Currently this will only fetch the second session of the film if there are two sessions
            for _,item in association.items():

                self.fmrifile = os.path.join(fmripath,item[1])

                ### load npz file 
                self.fmri = torch.FloatTensor(np.load(self.fmrifile)['X'])

                ### Check shape relative to other data types

                if self.fmri.shape[0] != self.audioset_proba.shape[0]:
                    #print("reshaping fmri and other data to minimum length of both")

                    min_len = min(self.fmri.shape[0],self.audioset_proba.shape[0])
                    self.fmri = self.fmri[:min_len,:]
                    self.audioset_proba = self.audioset_proba[:min_len,:]
                    self.im_proba = self.im_proba[:min_len,:]
                    self.places_proba = self.places_proba[:min_len,:]
                    self.onsets = self.onsets[:min_len]

        ### Prepare a 2D numpy array with all the wave chunks organized along the first dimension
        # 
        #  
        self.wavdata = []
        for onset in self.onsets:
            (chunk, _) = librosa.core.load(self.wavfile, sr=self.sample_rate, mono=True,offset=onset,duration=self.dur)
            self.wavdata.append(torch.FloatTensor(chunk))
        
    def __iter__(self):
        return iter(zip(self.wavdata,self.audioset_proba,self.im_proba,self.places_proba,self.fmri))

def train_kl(epoch,trainloader,net,optimizer,kl_im,kl_audio,kl_places,mseloss=None,alpha=1,beta=1,gamma=1,delta=1,epsilon=1):
    running_loss = 0
    net.train()

    for batch_idx, (onesample) in enumerate(trainloader):


        optimizer.zero_grad()
        bsize = onesample['waveform'].shape[0]

        

        # For fake 2D output
        #wav = torch.Tensor(onesample['waveform']).view(bsize,1,-1,1).cuda()
        #places = torch.Tensor(onesample['places']).view(bsize,-1,1,1).cuda()
        #audioset = torch.Tensor(onesample['audioset']).view(bsize,-1,1,1).cuda()
        #imnet = torch.Tensor(onesample['imagenet']).view(bsize,-1,1,1).cuda()

        # for 1D output
        wav = torch.Tensor(onesample['waveform']).view(bsize,1,-1,1).cuda()
        places = torch.Tensor(onesample['places']).view(bsize,1,-1).cuda()
        audioset = torch.Tensor(onesample['audioset']).view(bsize,1,-1).cuda()
        imnet = torch.Tensor(onesample['imagenet']).view(bsize,1,-1).cuda()

        # Forward pass
        obj_p,scene_p,audio_p,fmri_p = net(wav)

        # Calculate loss
        
        # For fake 2D output
        #loss_imagenet = kl_im(F.log_softmax(obj_p,2),imnet)
        #loss_audioset = kl_audio(F.log_softmax(audio_p,2),audioset)
        #loss_places = kl_places(F.log_softmax(scene_p,2),places)

        # For 1D output
        loss_imagenet = kl_im(F.log_softmax(obj_p,2),imnet)
        loss_audioset = kl_audio(F.log_softmax(audio_p,2),audioset)
        loss_places = kl_places(F.log_softmax(scene_p,2),places)
        
        if mseloss is not None:
            fmri = onesample['fmri'].view(bsize,1,-1).cuda()

            if net.maskattention is not None:
                
            
                masked_output = torch.matmul(fmri_p,net.maskattention)
                masked_target = torch.matmul(fmri,net.maskattention)

                lossattention = mseloss(masked_output,masked_target)

                lossortho = torch.norm(torch.matmul(net.maskattention.transpose(0,1),net.maskattention) - torch.eye(net.maskattention.shape[1]).cuda())

                loss_fmri= delta*lossattention + epsilon*lossortho
            else:
                loss_fmri=delta*mseloss(fmri_p,fmri)
            loss = alpha*loss_audioset + beta*loss_imagenet + gamma*loss_places + loss_fmri
        else:
            loss = alpha*loss_audioset + beta*loss_imagenet + gamma*loss_places

        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()
  
    return running_loss/batch_idx


def train(epoch,trainloader,net,optimizer,mseloss,delta=1e-2,epsilon=1e-4):
    all_fmri = []
    all_fmri_p = []
    running_loss = 0
    net.soundnet.eval()
    net.encoding_fmri.train()

    for batch_idx, (wav,audioset,imagenet,places,fmri) in enumerate(sample(trainloader,k=len(trainloader))):
        optimizer.zero_grad()
        bsize = wav.shape[0]

        # for 1D output
        #print(onesample['waveform'].shape)    
        wav = torch.Tensor(wav).view(1,1,-1,1).cuda() ### Major CHange here, inputting the wav for the whole batch at once in the network
        #print(wav.shape)     
        # Forward pass
        fmri_p = net(wav).permute(2,1,0,3).squeeze()
        #print(fmri_p.shape)
        
        #Cropping the end of the predicted fmri to match the measured bold
        fmri_p = fmri_p[:bsize]

        #print(fmri_p.shape)
        # Calculate loss
        
        fmri = fmri.cuda()
        #print(fmri.shape)
        all_fmri.append(fmri.cpu().numpy().reshape(bsize,-1))
        all_fmri_p.append(fmri_p.detach().cpu().numpy().reshape(bsize,-1))

        if net.maskattention is not None:
                
            
            masked_output = torch.matmul(fmri_p,net.maskattention)
            masked_target = torch.matmul(fmri,net.maskattention)

            lossattention = mseloss(masked_output,masked_target)

            lossortho = torch.norm(torch.matmul(net.maskattention.transpose(0,1),net.maskattention) - torch.eye(net.maskattention.shape[1]).cuda())

            loss= (delta*lossattention + epsilon*lossortho)/bsize
        else:
            loss=delta*mseloss(fmri_p,fmri)/bsize
            #loss=mseloss(fmri_p,fmri)/bsize  
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        

    r2_model = r2_score(np.vstack(all_fmri),np.vstack(all_fmri_p),multioutput='raw_values') 
    return running_loss/batch_idx, r2_model


def test(epoch,testloader,net,optimizer,mseloss,delta=1e-2,epsilon=1e-4):
    all_fmri = []
    all_fmri_p = []
    running_loss = 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (wav,audioset,imagenet,places,fmri) in enumerate(sample(testloader,k=len(testloader))):

            bsize = wav.shape[0]

            # load data
            wav = torch.Tensor(wav).view(1,1,-1,1).cuda()
            
            # Forward pass
            fmri_p = net(wav).permute(2,1,0,3).squeeze()



            #Cropping the end of the predicted fmri to match the measured bold
            fmri_p = fmri_p[:bsize]

            #print(fmri_p.shape)

            # Calculate loss

            # For 1D output
            
            fmri = fmri.view(bsize,-1).cuda()
            #print(fmri.shape)

            if net.maskattention is not None:
                masked_output = torch.matmul(fmri_p,net.maskattention)
                masked_target = torch.matmul(fmri,net.maskattention)

                lossattention = mseloss(masked_output,masked_target)

                lossortho = torch.norm(torch.matmul(net.maskattention.transpose(0,1),net.maskattention) - torch.eye(net.maskattention.shape[1]).cuda())

                loss= (delta*lossattention + epsilon*lossortho)/bsize
            else:
                loss=delta*mseloss(fmri_p,fmri)/bsize
                #loss = mseloss(fmri_p,fmri)/bsize
            
            all_fmri.append(fmri.cpu().numpy().reshape(bsize,-1))
            all_fmri_p.append(fmri_p[:bsize].cpu().numpy().reshape(bsize,-1))
            running_loss += loss.item()

        r2_model = r2_score(np.vstack(all_fmri),np.vstack(all_fmri_p),multioutput='raw_values')   
        return running_loss/batch_idx, r2_model

def train_map(epoch,trainloader,net,optimizer,mseloss,delta=1e-2,epsilon=1e-4):
    ## train function to use with the map-style dataloader
    all_fmri = []
    all_fmri_p = []
    running_loss = 0
    net.soundnet.eval()
    net.encoding_fmri.train()

    for batch_idx, (onesample, offset, duration) in enumerate((trainloader)):
        optimizer.zero_grad()
        bsize = onesample['waveform'].shape[0]

        # for 1D output
        #print(onesample['waveform'].shape)    
        wav = torch.Tensor(onesample['waveform']).view(1,1,-1,1).cuda() ### Major CHange here, inputting the wav for the whole batch at once in the network
        #print(wav.shape)     
        # Forward pass
        fmri_p = net(wav, offset, duration).permute(2,1,0,3).squeeze()
        #print(fmri_p.shape)
        
        #Cropping the end of the predicted fmri to match the measured bold
        fmri_p = fmri_p[:bsize]

        #print(fmri_p.shape)
        # Calculate loss
        
        fmri = onesample['fmri'].cuda()
        #print(fmri.shape)
        all_fmri.append(fmri.cpu().numpy().reshape(bsize,-1))
        all_fmri_p.append(fmri_p.detach().cpu().numpy().reshape(bsize,-1))

        if net.maskattention is not None:
                
            
            masked_output = torch.matmul(fmri_p,net.maskattention)
            masked_target = torch.matmul(fmri,net.maskattention)

            lossattention = mseloss(masked_output,masked_target)

            lossortho = torch.norm(torch.matmul(net.maskattention.transpose(0,1),net.maskattention) - torch.eye(net.maskattention.shape[1]).cuda())

            loss= (delta*lossattention + epsilon*lossortho)/bsize
        else:
            loss=delta*mseloss(fmri_p,fmri)/bsize
            #loss=mseloss(fmri_p,fmri)/bsize  
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        

    r2_model = r2_score(np.vstack(all_fmri),np.vstack(all_fmri_p),multioutput='raw_values') 
    return running_loss/batch_idx, r2_model


def test_map(epoch,testloader,net,optimizer,mseloss,delta=1e-2,epsilon=1e-4):
    ##test function to use with the map-style dataloader
    all_fmri = []
    all_fmri_p = []
    running_loss = 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (onesample, offset, duration) in enumerate(testloader):

            bsize = onesample['waveform'].shape[0]

            # load data
            wav = torch.Tensor(onesample['waveform']).view(1,1,-1,1).cuda()
            
            # Forward pass
            fmri_p = net(wav, offset, duration).permute(2,1,0,3).squeeze()



            #Cropping the end of the predicted fmri to match the measured bold
            fmri_p = fmri_p[:bsize]

            #print(fmri_p.shape)

            # Calculate loss

            # For 1D output
            
            fmri = onesample['fmri'].view(bsize,-1).cuda()
            #print(fmri.shape)

            if net.maskattention is not None:
                masked_output = torch.matmul(fmri_p,net.maskattention)
                masked_target = torch.matmul(fmri,net.maskattention)

                lossattention = mseloss(masked_output,masked_target)

                lossortho = torch.norm(torch.matmul(net.maskattention.transpose(0,1),net.maskattention) - torch.eye(net.maskattention.shape[1]).cuda())

                loss= (delta*lossattention + epsilon*lossortho)/bsize
            else:
                loss=delta*mseloss(fmri_p,fmri)/bsize
                #loss = mseloss(fmri_p,fmri)/bsize
            
            all_fmri.append(fmri.cpu().numpy().reshape(bsize,-1))
            all_fmri_p.append(fmri_p[:bsize].cpu().numpy().reshape(bsize,-1))
            running_loss += loss.item()

        r2_model = r2_score(np.vstack(all_fmri),np.vstack(all_fmri_p),multioutput='raw_values')   
        return running_loss/batch_idx, r2_model

def test_kl(epoch,testloader,net,optimizer,kl_im,kl_audio,kl_places,mseloss=None,alpha=1,beta=1,gamma=1,delta=1,epsilon=1):

    running_loss = 0
    net.eval()
    with torch.no_grad():
        for batch_idx, (onesample) in enumerate(testloader):

            bsize = onesample['waveform'].shape[0]

            # load data
            wav = torch.Tensor(onesample['waveform']).view(bsize,1,-1,1).cuda()
            places = torch.Tensor(onesample['places']).view(bsize,1,-1).cuda()
            audioset = torch.Tensor(onesample['audioset']).view(bsize,1,-1).cuda()
            imnet = torch.Tensor(onesample['imagenet']).view(bsize,1,-1).cuda()

            # Forward pass
            obj_p,scene_p,audio_p,fmri_p = net(wav)

            # Calculate loss
        
            # For fake 2D output
            #loss_imagenet = kl_im(F.log_softmax(obj_p,2),imnet)
            #loss_audioset = kl_audio(F.log_softmax(audio_p,2),audioset)
            #loss_places = kl_places(F.log_softmax(scene_p,2),places)

            # For 1D output
            loss_imagenet = kl_im(F.log_softmax(obj_p,2),imnet)
            loss_audioset = kl_audio(F.log_softmax(audio_p,2),audioset)
            loss_places = kl_places(F.log_softmax(scene_p,2),places)
            
            if mseloss is not None:
                fmri = onesample['fmri'].view(bsize,1,-1).cuda()


                if net.maskattention is not None:
                    
                
                    masked_output = torch.matmul(fmri_p,net.maskattention)
                    masked_target = torch.matmul(fmri,net.maskattention)

                    lossattention = mseloss(masked_output,masked_target)

                    lossortho = torch.norm(torch.matmul(net.maskattention.transpose(0,1),net.maskattention) - torch.eye(net.maskattention.shape[1]).cuda())

                    loss_fmri= delta*lossattention + epsilon*lossortho
                else:
                    loss_fmri=delta*mseloss(fmri_p,fmri)
                loss = alpha*loss_audioset + beta*loss_imagenet + gamma*loss_places + loss_fmri
            else:
                loss = alpha*loss_audioset + beta*loss_imagenet + gamma*loss_places
        
            

            running_loss += loss.item()
            
    return running_loss/batch_idx

def construct_dataloader(path, fmripath, audiopad,bsize=64):

    trainsets = []
    testsets= []
    valsets = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name[-3:] == 'mkv':
                currentvid = os.path.join(root, name)
                try:
                    dataset = AudioToEmbeddings(currentvid,fmripath=fmripath,samplerate=22050, audioPad=audiopad)
                    total_len = (len(dataset))
                    train_len = int(np.floor(0.6*total_len))
                    val_len = int(np.floor(0.2*total_len))
                    test_len = int(np.floor(0.2*total_len)) - 1

                    trainsets.append(torch.utils.data.Subset(dataset, range(train_len)))
                    valsets.append(torch.utils.data.Subset(dataset, range(train_len,train_len+val_len)))
                    testsets.append(torch.utils.data.Subset(dataset, range(train_len+val_len,train_len+val_len+test_len)))

                except FileNotFoundError as expr:
                    print("Issue with file {}".format(currentvid))
                    print(expr)
            
    trainset = torch.utils.data.ConcatDataset(trainsets)
    valset = torch.utils.data.ConcatDataset(valsets)
    testset = torch.utils.data.ConcatDataset(testsets)

    trainloader = DataLoader(trainset,batch_size=bsize)
    valloader = DataLoader(valset,batch_size=bsize)
    testloader = DataLoader(testset,batch_size=bsize)

    return trainloader, valloader, testloader, dataset


def construct_iter_dataloader(path, fmripath,bsize=10):

    datasets = []

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name[-3:] == 'mkv':
                currentvid = os.path.join(root, name)
                try:
                    dataset= AudioToEmbeddingsIterableDataset(currentvid,fmripath=fmripath,samplerate=22050)

                    datasets += (list(torch.utils.data.DataLoader(dataset,batch_size=bsize,drop_last=True)))

                    
                except FileNotFoundError as expr:
                    print("Issue with file {}".format(currentvid))
                    print(expr)

    ## Shuffle all the batches 

    datasets = sample(datasets,k=len(datasets))

    ### Divide in train val test

    total_len = len(datasets)  
    train_len = int(np.floor(0.6*total_len))
    val_len = int(np.floor(0.2*total_len))
    test_len = int(np.floor(0.2*total_len)) - 1

    trainloader = datasets[:train_len]
    valloader = datasets[train_len:train_len+val_len]
    testloader = datasets[train_len+val_len:train_len+val_len+test_len]
    return trainloader,valloader,testloader, dataset


    
    #trainloader = DataLoader(trainset,batch_size=bsize)
    #valloader = DataLoader(valset,batch_size=bsize)
    #testloader = DataLoader(testset,batch_size=bsize)

    #return trainloader, valloader, testloader, dataset
