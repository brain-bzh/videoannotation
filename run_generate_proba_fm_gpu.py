## Author : Nicolas Farrugia, February 2020
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import densenet161,resnet18
import torch
from torchvision.io import read_video,read_video_timestamps
import torchvision.transforms as transforms
import utils
import placesCNN_basic
import torch.nn as nn
from torch.nn import functional as F
from importlib import reload
from tqdm import tqdm
import os 
import sys
import numpy as np 

videofile = sys.argv[1]
wavfile = (videofile[:-3] + 'wav')
npzfile = (videofile[:-4] + '_fm_proba.npz')

from audioset_tagging_cnn.inference import audio_tagging

checkpoint_path='./LeeNet11_mAP=0.266.pth'

#### Check if audio file exists
if os.path.isfile(wavfile) is False:
    
    print('wav file does not exist, converting from {videofile}...'.format(videofile=videofile))

    utils.convert_Audio(videofile, wavfile)

#### If not, generate it and put it at the same place than the video file , as a wav, with the same name
#### use this following audio file to generate predictions on sound 

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# prepare the image transformer for Places Network
centre_crop = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
])

categories = utils.categories ### ImageNet Categories

places_categories= placesCNN_basic.classes ### Places Categories 

fps = 24
nb_frames = 1

nbsec = 1.49

n_obj = 3

beg_film = 1
end_film = 600

allpreds = []
onsets = []

model_imagenet = resnet18(pretrained=True).cuda()
model_imagenet.eval()

model_places = placesCNN_basic.model.cuda().eval()

### Define and register hook for extracting output feature map 
places_fm = []
places_proba = []

def get_fm_places(m, i, o):
    places_fm.append((i[0].cpu().numpy()[0]))

model_places.fc.register_forward_hook(get_fm_places)

im_fm = []
im_proba = []

def get_fm_im(m, i, o):
    im_fm.append((i[0].cpu().numpy()[0])) 

model_imagenet.fc.register_forward_hook(get_fm_im)

audioset_fm = []
audioset_proba = []

with torch.no_grad():    
    for curstart in tqdm(np.arange(beg_film,end_film,nbsec)):

        start = curstart
        end = start + (nb_frames/fps)

        onsets.append(curstart)

        
        vframes, aframes, info = read_video(filename=videofile,start_pts = start,end_pts=end,pts_unit='sec')

        vframes = vframes.permute(0,3,1,2).float() / 255

        _,H,C,V = vframes.shape


        ### make prediction for Places 

        im_norm = centre_crop(vframes[0]).reshape(1,3,224,224).cuda()
        preds_places = model_places(im_norm)

        ### make prediction for Imagenet classification 

        #im_norm = normalize(vframes[0]).reshape(1,H,C,V)
        
        preds_class= model_imagenet(im_norm)
        # Make predictions for audioset 
        clipwise_output, labels,sorted_indexes,embedding = audio_tagging(wavfile,checkpoint_path,offset=curstart,duration=nbsec,usecuda=True)

        audioset_fm.append(embedding)

        ### Associate Classification labels to ImageNet prediction 

        allclasses = preds_class.data.cpu().numpy()[0]

        # process output of Imagenet Classes and print results:
        order = allclasses.argsort()
        last = len(categories)-1
        

        proba_im = F.softmax(preds_class.data[0], 0).data.squeeze()
        im_proba.append(proba_im.cpu().numpy())
        
        # process output of Places Classes and print results:

        _, idx = preds_places[0].sort(0, True)

        places_proba.append(F.softmax(preds_places, 1).data.squeeze().cpu().numpy())

        ## AUdioSet
        audioset_proba.append(clipwise_output)

       
## Removing temporary wave file 
os.remove(wavfile)

### Saving feature maps, probabilities and metadata
np.savez_compressed(npzfile,places_fm = np.stack(places_fm),im_fm = np.stack(im_fm),
    audioset_fm=np.stack(audioset_fm),
    places_proba = np.stack(places_proba),audioset_proba=np.stack(audioset_proba),im_proba=np.stack(im_proba),
    dur=nbsec,onsets=onsets)