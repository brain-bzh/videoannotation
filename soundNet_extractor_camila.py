import matplotlib.pyplot as plt

#utility 
import os
import numpy as np
import pickle

#audio
import librosa

#network
import soundnet_extract_features as ex

def extract_vector(features,layer):
    # C : channel_output and H : Height
    # Vector's shape (H,C)
    C = features[layer][0].shape[0]    
    print("Cout : ",C)
    return (features[layer].reshape(C,-1).T)

def extract_vectors(filepath):
    audio,sr = ex.load_audio(filepath)
    features = ex.extract_pytorch_feature(audio,'./sound8.pth')
   
    print([x.shape for x in features])
    
    ##extract vector
    vectors = []
    for idlayer in range(len(features)):
        vectors.append(ex.extract_vector(features,idlayer)) #features vector 
    return vectors, len(audio)/sr


wavfile = '/media/brain/Elec_HD/cneuromod/movie10/stimuli/wolf_of_wall_street/the_wolf_of_wall_street_seg16_subsampl.wav'   
vector,size = extract_vectors(wavfile)

print(f'shape : ', vector[6].shape)
print(f'size : ', size)

plt.imshow(vector[6])
plt.show()