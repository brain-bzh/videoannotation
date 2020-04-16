
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nistats import hemodynamic_models

def experimental_matrix(trials_length, inter_trials_interval, stimuli_numbers, stimuli_duration, inter_stimuli_intervals, tr = 1, amplitude_stimuli = 1):
    total_length = sum(trials_length)+(len(trials_length)+1)*inter_trials_interval
    frame_times = np.arange(start = 0, stop = total_length, step = tr)

    exp_conditions = []
    exp_stimuli = []
    for stim_nb, ISI, duration in zip (stimuli_numbers, inter_stimuli_intervals, stimuli_duration) : 
        onsets = []
        for i, trial in enumerate(trials_length) : 
            onsets_trial = np.linspace(0, trial, stim_nb)+(inter_trials_interval*(i+1)+sum(trials_length[:i]))
            onsets = np.concatenate((onsets, onsets_trial))
        onsets = np.round(onsets.reshape(onsets.shape[0], 1), decimals=2)

        durations = []
        for i in duration :
            dur = np.ones(stim_nb)
            dur = dur * i
            durations = np.concatenate((durations, dur))
        durations = np.round(durations.reshape(durations.shape[0], 1), decimals=2)

        amplitudes = []
        if isinstance(amplitude_stimuli, (int, float)) :
            amplitudes = np.ones_like(onsets)*amplitude_stimuli
        elif isinstance(amplitude_stimuli, (list, np.ndarray)):
            amplitudes = np.array(amplitude_stimuli)
        amplitudes = amplitudes.reshape(amplitudes.shape[0], -1)

        if onsets.shape[1] != amplitudes.shape[1]:
            onsets = np.repeat(onsets, amplitudes.shape[1], axis = 1)
        if durations.shape[1] != amplitudes.shape[1]:
            durations = np.repeat(durations, amplitudes.shape[1], axis = 1)

        print(f'onsets shape : ', onsets.shape)
        print(f'amplitudes shape : ', amplitudes.shape)
        print(f'durations shape : ', durations.shape)

        exp_cdt = (onsets, durations, amplitudes)
        exp_conditions.append(exp_cdt)

        stimuli = np.zeros_like(frame_times)
        for start, duration in zip(onsets[:,0], durations[:,0]):
            stimuli[int(start):int(start+duration)] = 1
        exp_stimuli.append(stimuli)

    return exp_conditions, exp_stimuli, frame_times


if __name__ == "__main__":
    #conditions of exp
    tr = 1
    trials_length = [2,4,8,15,30]
    inter_trials_interval = 15

    #stimuli definitions
    stimuli_numbers = [1, 30, 30]
    inter_stimuli_intervals = [0, [0.033, 0.100, 0.233, 0.467, 0.965], 0]
    stimuli_duration = [trials_length, [0.033, 0.033, 0.033, 0.033, 0.033], [0.067,0.133,0.267, 0.500, 1.000]]

    exp_conditions, exp_stimuli, frame_times = experimental_matrix(trials_length, inter_trials_interval, stimuli_numbers, stimuli_duration, inter_stimuli_intervals, tr, amplitude_stimuli=1)

    hrf_model = 'spm'
    oversampling = 16
    index = 2

    stim_nb, exp_condition, stim = stimuli_numbers[index], exp_conditions[index], exp_stimuli[index]

    fig = plt.figure(figsize=(30,4))

    signal, name = hemodynamic_models.compute_regressor(exp_condition, 
                        hrf_model, frame_times, con_id='main', oversampling=oversampling)
    plt.subplot(1,1,1)
    plt.fill(frame_times, stim, 'k', alpha=0.5, label='stimulus')
        
    for j in range(signal.shape[1]):
        plt.plot(frame_times, signal.T[j], label=name[j])
    plt.xlabel('time (ms)')
    plt.legend(loc=1)
    plt.title(hrf_model)    

    plt.subplots_adjust(bottom=.12)
    plt.savefig('test_{}.jpg'.format(index))