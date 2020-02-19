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
srtfile = (videofile[:-3] + 'srt')
wavfile = (videofile[:-3] + 'wav')
npzfile = (videofile[:-4] + '_fm.npz')

from audioset_tagging_cnn.inference import audio_tagging

checkpoint_path='./LeeNet11_mAP=0.266.pth'

#### Check if audio file exists
if os.path.isfile(wavfile) is False:
    
    print('wav file does not exist, converting from {videofile}...'.format(videofile=videofile))

    utils.convert_Audio(videofile, wavfile)

#### If not, generate it and put it at the same place than the video file , as a wav, with the same name
#### use this following audio file to generate predictions on sound 


if os.path.isfile(srtfile):
    print('Removing exisiting subtitles file {file}'.format(file=srtfile))
    os.remove(srtfile)

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

nbsec = 1

n_obj = 3

beg_film = 1
end_film = 600

allpreds = []
onsets = []

model_imagenet = resnet18(pretrained=True)
model_imagenet.eval()

model_places = placesCNN_basic.model.eval()

### Define and register hook for extracting output feature map 
places_fm = []
places_proba = []

def get_fm_places(m, i, o):
    places_fm.append((i[0].numpy()[0]))

model_places.fc.register_forward_hook(get_fm_places)

im_fm = []
im_proba = []

def get_fm_im(m, i, o):
    im_fm.append((i[0].numpy()[0])) 

model_imagenet.fc.register_forward_hook(get_fm_im)

audioset_fm = []
audioset_proba = []

n=0
with torch.no_grad():    
    for curstart in tqdm(range(beg_film,end_film,nbsec)):

        start = curstart
        end = start + (nb_frames/fps)

        onsets.append(curstart)

        
        vframes, aframes, info = read_video(filename=videofile,start_pts = start,end_pts=end,pts_unit='sec')

        vframes = vframes.permute(0,3,1,2).float() / 255

        _,H,C,V = vframes.shape


        ### make prediction for Places 

        im_norm = centre_crop(vframes[0]).reshape(1,3,224,224)
        preds_places = model_places(im_norm)

        ### make prediction for Imagenet classification 

        #im_norm = normalize(vframes[0]).reshape(1,H,C,V)
        
        preds_class= model_imagenet(im_norm)


        # Make predictions for audioset 
        clipwise_output, labels,sorted_indexes,embedding = audio_tagging(wavfile,checkpoint_path,offset=curstart,duration=nbsec)

        audioset_fm.append(embedding)

        ### Associate Classification labels to ImageNet prediction 

        allclasses = preds_class.data.numpy()[0]

        # process output of Imagenet Classes and print results:
        order = allclasses.argsort()
        last = len(categories)-1
        text = ''
        for i in range(min(3, last+1)):
            text += categories[order[last-i]]
            text += ', '
        text=text[:-2]

        proba_im = F.softmax(preds_class.data[0], 0).data.squeeze()
        im_proba.append(proba_im)
        print(proba_im)

        # process output of Places Classes and print results:

        _, idx = preds_places[0].sort(0, True)

        textplaces = ''
        for i in range(0, 5):
            textplaces += places_categories[idx[i]]
            textplaces += ', '
        textplaces = textplaces[:-2]

        places_proba.append(F.softmax(preds_places, 1).data.squeeze())

        # Print audio tagging top probabilities
        texttagging = ''
        for k in range(3):
            texttagging += np.array(labels)[sorted_indexes[k]]
            texttagging += ', '
            print(clipwise_output[sorted_indexes[k]])
        texttagging = texttagging[:-2]

        audioset_proba.append(clipwise_output)
            
        ### Generate final string

        annotation_str = "Audioset: {tagging}\nPLACES: {places}\nImageNet : {net}".format(tagging=texttagging,places=textplaces,net=text)

        #print(annotation_str)

        ### Append to srt file with timecode 
        utils.gen_srt(annotation_str,start,srtfile=srtfile,num=n,duration=nbsec)
        n=n+1
        
## Removing temporary wave file 
os.remove(wavfile)

### Saving feature maps, probabilities and metadata
np.savez_compressed(npzfile,places_fm = np.stack(places_fm),im_fm = np.stack(im_fm),
    audioset_fm=np.stack(audioset_fm),
    places_proba = places_proba,audioset_proba=audioset_proba,im_proba=im_proba,
    dur=nbsec,onsets=onsets)