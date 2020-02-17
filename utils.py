## Author : Nicolas Farrugia, February 2020

from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
from torchvision.io import read_video,read_video_timestamps

import matplotlib.patches as patches
from matplotlib import pyplot as plt

import datetime

import os



def convert_Audio(mediaFile, outPath, outExtension = '.wav', name = None):
    if name == None : 
        name, extension = os.path.splitext(os.path.basename(mediaFile))
    outFile = os.path.join(outPath, name+outExtension)
    cmd = 'ffmpeg -i '+mediaFile+' '+outFile
    os.system(cmd)
    return outFile


#### imagenet categories

def cat_file():
    # load classes file
    categories = []
    try:
        f = open('categories.txt', 'r')
        for line in f:
            cat = line.split(',')[0].split('\n')[0]
            if cat != 'classes':
                categories.append(cat)
        f.close()
        print('Number of categories:', len(categories))
    except:
        print('Error opening file ' + ' categories.txt')
        quit()
    return categories


categories = cat_file() # load category file




def annotate_img(preds,vframes,n_obj=5):

    ### vframes  : last three dims input tensor to faster_rcnn

    ### preds : dictionary of outputs of fater_rcnn

    global COCO_INSTANCE_CATEGORY_NAMES

    predlabels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in preds['labels'].numpy()]
    scores = [i for i in preds['scores'].detach().numpy()]
    bboxes = [i for i in preds['boxes'].detach().numpy()]


    test_im = vframes.permute(1,2,0).numpy()

    # Create figure and axes
    fig,ax = plt.subplots(1,figsize=(20,25))

    # Display the image
    ax.imshow(test_im)


    #### add the annotations 

    for curbbox,curlab in zip(bboxes[:n_obj],predlabels[:n_obj]):

        topleftx = curbbox[0]
        toplefty = curbbox[1]

        bottomrightx = curbbox[2]
        bottomrighty = curbbox[3]


        # Create a Rectangle patch
        rect = patches.Rectangle((topleftx,toplefty),abs(bottomrightx-topleftx),abs(bottomrighty-toplefty),linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        ax.text(topleftx,toplefty,curlab,c='white',fontsize=16)
    plt.show()
    return fig


def gen_srt(strlabel,onset,srtfile,duration=2,num=1):

    starttime = onset
    endtime = starttime + duration

    string_start = datetime.time(0,starttime//60,starttime%60).strftime("%H:%M:%S")

    string_end = datetime.time(0,endtime//60,endtime%60).strftime("%H:%M:%S")

    with open(srtfile,'a') as f:
        f.write("{}\n".format(num+1))
        f.write("{starttime} --> {endtime}\n".format(starttime=string_start,endtime=string_end))
        f.write("{}\n".format(strlabel))
        f.write("\n")

def gen_srt_coco_multiple(allpreds,onsets,srtfile,n_obj=5):
    global COCO_INSTANCE_CATEGORY_NAMES

    ## check that both lists have the same size 
    if len(allpreds) != len(onsets):
        raise(ValueError('List of predictions and onsets have different sizes'))

    for num,(curpred,curonset) in enumerate(zip(allpreds,onsets)):

        predlabels = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in curpred['labels'].numpy()[:n_obj]]

        starttime = curonset
        endtime = curonset + 2

        string_start = datetime.time(0,starttime//60,starttime%60).strftime("%H:%M:%S")

        string_end = datetime.time(0,endtime//60,endtime%60).strftime("%H:%M:%S")

        with open(srtfile,'a') as f:
            f.write("{}\n".format(num+1))
            f.write("{starttime} --> {endtime}\n".format(starttime=string_start,endtime=string_end))
            f.write("{}\n".format(predlabels))
            f.write("\n")
