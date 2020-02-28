## Author : Nicolas Farrugia, February 2020
from matplotlib import pyplot as plt 
import torch
import torch.nn as nn
from tqdm import tqdm
import os 
import sys
import numpy as np 

from nilearn.plotting import plot_stat_map

from nilearn.regions import signals_to_img_labels

import sys

""" 
state = {
                                    'net': net.state_dict(),
                                    'epoch': epoch,
                                    'train_loss' : train_loss,
                                    'val_loss' : val_loss,
                                    'test_loss' : test_loss,
                                    'nfeat' : nfeat,
                                    'ninputfilters' : ninputfilters,
                                    'model' : net
                                } """

path = sys.argv[1]

for root, dirs, files in os.walk(path, topdown=False):
   for name in files:
       if name[-2:] == 'pt':
           currentdict = os.path.join(root, name)
           print(currentdict)
           try:
               test_loss=torch.load(currentdict)['test_loss']
               print(test_loss)
           except:
               print('Error')

           

