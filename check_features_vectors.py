import numpy as np
import torch
import torch.nn as nn
import librosa
from soundnet_model import SoundNet8_pytorch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

path_FV_Layer7 = '/home/brain/git/neuromod-movie10-encoding/FV-soundnet/features_vectors_conv_7/bourne_supremacy_seg10.npz'
wav_file = '/media/brain/Elec_HD/cneuromod/movie10/stimuli/bourne_supremacy/bourne_supremacy_seg10_subsampl.wav'
proba_file = '/media/brain/Elec_HD/cneuromod/movie10/stimuli/bourne_supremacy/bourne_supremacy_seg10_fm_proba.npz'
pytorch_param_path='./sound8.pth'

net = SoundNet8_pytorch()
net.load_state_dict(torch.load(pytorch_param_path))
gpool = nn.AdaptiveAvgPool2d((1,1))
#gpool = nn.AvgPool2d((2,1))

fv_layer7 = np.load(path_FV_Layer7)['x']
print(fv_layer7.shape)

dur = np.load(proba_file)['dur']
onsets = np.load(proba_file)['onsets']
print(onsets[:-1].shape, dur)

#all_fv_wav = np.array([]).reshape((0,1024))

# for i, offset in enumerate(onsets[:-1]) : 
#     (waveform, _) = librosa.core.load(wav_file, sr=22050, mono=True,offset=offset,duration=dur)
#     wav = torch.Tensor(waveform).view(1,1,-1,1)
#     fv_wav = net(wav)
#     fv_wav = gpool(fv_wav).view(1024)
#     fv_wav = fv_wav.detach().numpy()
#     fv_wav = fv_wav.reshape((1, -1))
#     all_fv_wav = np.concatenate((all_fv_wav, fv_wav))
#     #if i%20 == 0:
#         #matrix = cosine_similarity(fv_layer7[i].reshape(1,-1), fv_wav)
#         #plt.matshow(matrix)
#         #plt.show()

(waveform, _) = librosa.core.load(wav_file, sr=22050, mono=True)

wav = torch.Tensor(waveform).view(1,1,-1,1)
fv_wav = net(wav)
fv_wav = fv_wav.view(-1, 1024)
all_fv_wav = fv_wav.detach().numpy()
#all_fv_wav = fv_wav.reshape((-1,1024))

print(all_fv_wav.shape)

plt.imshow(all_fv_wav)
plt.show()

# matrix = cosine_similarity(fv_layer7, all_fv_wav)
# plt.figure()
# plt.matshow(matrix)
# plt.colorbar()
# plt.title('2 fv')


# matrix2 = cosine_similarity(fv_layer7)
# plt.figure()
# plt.matshow(matrix2)
# plt.colorbar()
# plt.title('fv_layer7')


# matrix3 = cosine_similarity(all_fv_wav)
# ind = np.diag_indices_from(matrix3)
# matrix3[ind] = 0

# plt.figure()
# plt.matshow(matrix3)
# plt.colorbar()
# plt.title('fv_wav')
# plt.show()





# print(all_fv_wav.shape)

# matrixs = [cosine_similarity(fv_layer7, all_fv_wav), cosine_similarity(fv_layer7), cosine_similarity(all_fv_wav)]
# titles = ['2 fv', 'fv_layer7', 'fv_wav']
# fig,axs = plt.subplots(1,3)

# for col in range(3):
#     ax = axs[col]
#     ax.matshow(matrixs[col])
#     plt.title(titles[col])

# plt.show()




