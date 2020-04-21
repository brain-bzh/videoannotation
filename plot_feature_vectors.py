import numpy as np
import torch
import torch.nn as nn
import librosa
from soundnet_model import SoundNet8_pytorch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

path_FV_Layer7 = '/home/brain/git/neuromod-movie10-encoding/FV-soundnet/features_vectors_conv_7/the_wolf_of_wall_street_seg16.npz'

fv_layer7 = np.load(path_FV_Layer7)['x']
print(fv_layer7.shape)


# X = range(1,405)
# plt.plot(X, fv_layer7[:, 505])
# plt.legend(505)
# plt.show()

plt.imshow(fv_layer7)
plt.show()
