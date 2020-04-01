#-*- coding: utf-8 -*-
# @Author: XXXXXX
# Test roalway and school
# based on : https://github.com/smallflyingpig/SoundNet_Pytorch 

import argparse

#load with soundfile
import soundfile as sf
import time
import matplotlib.pyplot as plt 
import numpy as np 
#add relative
from soundnet_model import * 

import librosa

def extract_pytorch_feature(input_data:np.ndarray, pytorch_param_path:str)->list:
    import torch
    # "point invalid" error, if put the import on the top of this file
    model = SoundNet8_pytorch()
    # load model
    model.load_state_dict(torch.load(pytorch_param_path))
    # convert data to tensor
    data = torch.from_numpy(input_data).view(1,1,-1,1)
    print('Tensor shape:',data.shape)
    model.eval()
    with torch.no_grad():
        feature_all = model.extract_feat(data)
    return feature_all


def extract_pytorch_feature_nooutput(input_data:np.ndarray, pytorch_param_path:str)->list:
    import torch
    # "point invalid" error, if put the import on the top of this file
    model = SoundNet8_pytorch()
    # load model
    model.load_state_dict(torch.load(pytorch_param_path))
    # convert data to tensor
    data = torch.from_numpy(input_data).view(1,1,-1,1)
    #print('Tensor shape:',data.shape)
    model.eval()
    with torch.no_grad():
        feature_all = model.extract_feat_nooutput(data)
    return feature_all


def get_parser():
    parser = argparse.ArgumentParser("check")
    parser.add_argument("--input_param_path", type=str, default="../audio22k/railroad_audio_.wav")
    parser.add_argument("--pytorch_param_path", type=str, default="./sound8.pth")
    args, _ = parser.parse_known_args()
    return args

def load_audio(filepath):
    start = time.time()
    #sound, sr = sf.read(filepath,dtype ='float32',samplerate=22050)      
    sound,sr = librosa.load(filepath,sr=22050)
    if sr != 22050:
        print("Sampling rate dif, making")
        # TODO os.system()
        return
    print('time: ', time.time()-start)
    return sound, sr
    
def vector_to_scenes(scenes_vector):
    f = open("../sound/categories_places2.txt",'r')
    categories = f.read().split("\n")
    for x in range(len(scenes_vector[0])):
        index = np.argmax(scenes_vector[:,x])
        print(categories[index])

def vector_to_obj(obj):
    f = open("../sound/objects.txt",'r')
    categories = f.read().split("\n")
    for x in range(len(obj[0])):
        index = np.argmax(obj[:,x])
        print(categories[index])

def extract_vector(features,layer):
    # C : channel_output and H : Height
    # Vector's shape (H,C)
    C = features[layer][0].shape[0]    
    #print("Cout : ",C)
    return (features[layer].reshape(C,-1).T)

def get_values():
        to_time = {}
        with open('./relation_layer_seconds.txt', 'r') as reader:
            for i in reader:
                key,m,b = i.split()
                if key != 'name_layer':
                    to_time[key] = [float(m),float(b)]
        return to_time


def main(args):        
    input_data,sr = load_audio(args.input_param_path)
    print("extrach features using pytorch model...")
    feature_all_pytorch = extract_pytorch_feature(input_data, args.pytorch_param_path)
    scenes_vector = extract_vector(feature_all_pytorch,8) #extract last layer
    vector_to_scenes(scenes_vector)


   
# test
if __name__=="__main__":
    args = get_parser()
    main(args)
