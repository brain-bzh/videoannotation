## Author : Nicolas Farrugia, February 2020
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import densenet161
import torch
from torchvision.io import read_video,read_video_timestamps
import torchvision.transforms as transforms
import objectdetection
import placesCNN_basic
import torch.nn as nn
from importlib import reload
from tqdm import tqdm
import os 
import sys

videofile = sys.argv[1]
srtfile = (videofile[:-3] + 'srt')
wavfile = (videofile[:-3] + 'wav')

#### TO DO 
#### Check if audio file exists
#### If not, generate it and put it at the same place than the video file , as a wav, with the same name
#### use this following audio file to generate predictions on sound 
if os.path.isfile(wavfile) is False:
    
    raise(NotImplementedError('wav file does not exist, please convert from {videofile}'.format(videofile=videofile)))


if os.path.isfile(srtfile):
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

categories = objectdetection.categories ### ImageNet Categories

places_categories= placesCNN_basic.classes ### Places Categories 

### Define and register hook for extracting output feature map 
#visualisation = []

#def hook_fn(m, i, o):
#    visualisation.append(o) 

#model.roi_heads.box_predictor.cls_score.register_forward_hook(hook_fn)

fps = 24
nb_frames = 1

nbsec = 3

n_obj = 3

beg_film = 1
end_film = 600

allpreds = []
onsets = []

model_imagenet = densenet161(pretrained=True)
model_imagenet.eval()

model_places = placesCNN_basic.model.eval()


n=0

with torch.no_grad():    
    for curstart in tqdm(range(beg_film,end_film,nbsec)):

        start = curstart
        end = start + (nb_frames/fps)

        vframes, aframes, info = read_video(filename=videofile,start_pts = start,end_pts=end,pts_unit='sec')

        vframes = vframes.permute(0,3,1,2).float() / 255

        _,H,C,V = vframes.shape


        ### make prediction for Places 

        im_norm = centre_crop(vframes[0]).reshape(1,3,224,224)
        preds_places = model_places(im_norm)

        ### make prediction for Imagenet classification 

        #im_norm = normalize(vframes[0]).reshape(1,H,C,V)
        
        preds_class= model_imagenet(im_norm)

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


        # process output of Places Classes and print results:

        _, idx = preds_places[0].sort(0, True)

        textplaces = ''
        # output the prediction
        for i in range(0, 5):
            textplaces += places_categories[idx[i]]
            textplaces += ', '
        textplaces = textplaces[:-2]

        ### Generate final string

        annotation_str = "PLACES: {places}\nImageNet : {net}".format(places=textplaces,net=text)

        #print(annotation_str)

        ### Append to srt file with timecode 
        objectdetection.gen_srt(annotation_str,start,srtfile=srtfile,num=n)
        n=n+1