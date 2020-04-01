import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_utils import AudioToEmbeddings

def construct_dataloader(path, fmripath, audiopad):

    trainsets = []
    testsets= []
    valsets = []

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name[-3:] == 'mkv':
                currentvid = os.path.join(root, name)
                #print(currentvid)
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

    trainloader = DataLoader(trainset,batch_size=64,shuffle=True)
    valloader = DataLoader(valset,batch_size=64)
    testloader = DataLoader(testset,batch_size=64)

    return trainloader, valloader, testloader, dataset