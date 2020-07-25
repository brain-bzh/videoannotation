 
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from soundnet_model import SoundNet8_pytorch
from MRI_utils import experimental_matrix
from nistats import hemodynamic_models
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise


def load_net_pretrained(pytorch_net, params_path, cuda=True, num_cuda=0):
    '''create a pytorch model and load the corresponding parameters. 
    If cuda = True, move the model to the cuda device.

    return the pretrained model.

    input : 
        pytorch_net [pytorch class]: the ML model
        params_path [string]: path to the file containing the trained parameters of the model
        cuda [boolean]: decision to use cuda to accelerate the computations
        num_cuda [int]: numero of the cuda device used for the computations

    output: the instanciated object net.
    '''
    net = pytorch_net()
    net.load_state_dict(torch.load(params_path))
    if cuda : 
        net = net.cuda(num_cuda)
    return net


def SoundNet_FV_from_audio(path, SoundNet8_param_path, sample_rate=22050, tr=1, audiopad=0):

    soundnet = load_net_pretrained(SoundNet8_pytorch, SoundNet8_param_path, cuda=True, num_cuda = 0)
    gpool = nn.AdaptiveAvgPool2d((1,1))

    times_FV = None
    audio_lentgh = librosa.core.get_duration(filename = wavfile)
    for offset in np.arange(start = 0, stop = audio_lentgh-(tr*2*audiopad+1), step = tr) :
        #suppress last/incomplete segment from audio
        if audio_lentgh-offset >= tr:
            (waveform, _) = librosa.core.load(wavfile, sr=sample_rate, mono = True, offset=offset ,duration=tr*(2*audiopad+1))
            x = np.reshape(waveform, (1,1,-1,1))
            x = torch.from_numpy(x).cuda(0)
            y = soundnet(x).cpu()
            y = gpool(y)
            y = y.view(1, -1).detach().numpy()
            if times_FV is None : 
                times_FV = y
            else : 
                times_FV = np.concatenate((times_FV, y), 0)
    return times_FV

def audio_subsampling_by_tr(wavfile, subsampling = 'first', tr=1.49):
    subsamples = []
    audio_length = librosa.core.get_duration(filename = wavfile)
    for offset in np.arange(start = 0, stop = audio_length, step = tr) :
        if audio_length-offset >= tr:
            (waveform, _) = librosa.core.load(wavfile, sr=sample_rate, mono = True, offset=offset ,duration=tr)
            if subsampling == 'first':
                subsamples.append(waveform[0])
            elif subsampling == 'max':
                subsamples.append(max(waveform))
            elif subsampling == 'mean':
                subsamples.append(np.mean(waveform))
    return subsamples
                

def show_similarity_matrix(save_path, X, label_X, Y = None, label_Y = None) : 
    if label_Y == None:
        label_Y = label_X
    matrix = pairwise.cosine_similarity(X, Y)
    plt.matshow(matrix)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar()
    plt.title('{} X {}'.format(label_X, label_Y))
    plt.savefig(os.path.join(save_path, '{} X {}'.format(label_X, label_Y)))
    plt.close()
    return matrix


if __name__ == "__main__":
    sample_rate = 22050
    tr = 1.49
    path = '/home/maelle/Database/cneuromod/movie10/stimuli/life'
    result_path = '/home/maelle/Results/test_hrf_life'
    SoundNet8_param_path = './sound8.pth'
    
    for file1 in os.listdir(path):
        name, ext = os.path.splitext(file1)
        if ext == ".wav":
            save_path = os.path.join(result_path, name)
            if not os.path.isdir(os.path.join(result_path, name)):
                os.mkdir(save_path)
            for audiopad in range(4):
                wavfile = os.path.join(path, file1)
                audio_length = librosa.core.get_duration(filename = wavfile)

                #FV matrix
                times_FV = SoundNet_FV_from_audio(wavfile, SoundNet8_param_path, sample_rate, tr, audiopad)
                show_similarity_matrix(save_path = save_path, X = times_FV, label_X = 'audiopad{} FV'.format(audiopad))

                #Stimuli_amplitudes_matrix
                first_audioframe = audio_subsampling_by_tr(wavfile, subsampling = 'first', tr=1.49)
                max_audioframe = audio_subsampling_by_tr(wavfile, subsampling = 'max', tr=1.49)
                mean_audioframe = audio_subsampling_by_tr(wavfile, subsampling = 'mean', tr=1.49)
                #stimuli_origins = ['first audio frame', 'maximum of the audio frames', 'mean of the audio frames']
               
                #experimental conditions matrix + hrf regressors
                tr = 1.49
                trials_length = [audio_length-(tr*2*audiopad)]
                inter_trials_interval = 0

                stimuli_amplitude = times_FV
                stimuli_numbers = len(stimuli_amplitude)
                stimuli_duration = [tr]
                inter_stimuli_intervals = np.zeros_like(stimuli_amplitude)
                
                (onsets, durations, amplitudes), frame_times = experimental_matrix(trials_length, inter_trials_interval, stimuli_numbers, stimuli_duration, inter_stimuli_intervals, tr, stimuli_amplitude)

                hrf_models = [None, 'spm', 'glover']
                oversampling = 16
                for i, hrf_model in enumerate(hrf_models) :
                    all_features = []
                    for onset, duration, amplitude in zip(onsets.T, durations.T, amplitudes.T):
                        exp_conditions = (onset, duration, amplitude)
                        signal, leg = hemodynamic_models.compute_regressor(exp_conditions, hrf_model, frame_times, oversampling=oversampling)
                        all_features.append(signal[:-1])
                    HRF_features = np.squeeze(np.stack(all_features)).T
                    print(HRF_features.shape, times_FV.shape)
                    show_similarity_matrix(save_path, times_FV, 'audiopad {} : features vector'.format(audiopad), HRF_features, hrf_model)


                # matrix_similarity = {}
                # for stimuli_amplitude, stimuli_origin in zip([first_audioframe, max_audioframe, mean_audioframe], stimuli_origins):
                #     stimuli_numbers = [len(stimuli_amplitude)]
                #     stimuli_duration = [[tr]]
                #     inter_stimuli_intervals = np.zeros_like(stimuli_amplitude)
                    
                #     [exp_conditions], [exp_stimuli], frame_times = experimental_matrix(trials_length, 
                #     inter_trials_interval, stimuli_numbers, stimuli_duration, inter_stimuli_intervals, 
                #     tr, stimuli_amplitude)

                #     models_signals=[]
                #     fig1 = plt.figure(figsize=(30,4*len(hrf_models)))
                #     matrix = {}
                #     for i, hrf_model in enumerate(hrf_models) :
                #         signal, leg = hemodynamic_models.compute_regressor(exp_conditions, 
                #         hrf_model, frame_times, oversampling=oversampling)
                #         models_signals.append(signal)

                #         stimuli = np.array(stimuli_amplitude).reshape(-1,1)
                #         matrix_HRF = pairwise.cosine_similarity(stimuli, signal[:-1])
                #         matrix[hrf_model] = matrix_HRF

                #         #figures_HRF
                #         plt.subplot(len(hrf_models), 1, i + 1)
                #         for j in range(signal.shape[1]):
                #             plt.plot(frame_times, signal.T[j], label=leg[j])
                #         plt.xlabel('time (ms)')
                #         plt.legend(loc=1)
                #         plt.title(hrf_model)

                #     fig1.tight_layout(pad=3.0)
                #     #fig.title('results for the {} in a audio segment of {}s'.format(stimuli_origin, tr))
                #     save_name = os.path.join(save_path, '{}_results_for_the_{}.jpg'.format(name, stimuli_origin))    
                #     plt.savefig(save_name)
                #     plt.close()

                #     matrix_similarity[stimuli_origin] = matrix
            
                # for stim, dico in matrix_similarity.items():
                #     for hrf_model, matrix in dico.items():
                #         show_similarity_matrix(save_path, matrix, stim, hrf_model)

            