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

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
] ### COCO Categories

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

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

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

        im_norm = normalize(vframes[0]).reshape(1,H,C,V)
        
        preds_class= model_imagenet(im_norm)

        ### make predictions for object detection

        preds = model(vframes)

        ### Associate Detection labels to prediction and keep only the first n_obj

        predlabels_det = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in preds[0]['labels'].numpy()[:n_obj]]

        ### Associate Classification labels to prediction 

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

        annotation_str = "PLACES: {places} \nCOCO : {coco} \nImageNet : {net}".format(places=textplaces,coco=str(predlabels_det),net=text)

        #print(annotation_str)

        ### Append to srt file with timecode 
        objectdetection.gen_srt(annotation_str,start,srtfile=srtfile,num=n)
        n=n+1